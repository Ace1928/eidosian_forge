import base64
import binascii
import collections
import copy
import json
from typing import List, Optional, Text
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_digest
from containerregistry.client.v2_2 import docker_http
from containerregistry.client.v2_2 import docker_image
from containerregistry.client.v2_2 import docker_session
from containerregistry.transform.v2_2 import metadata
from googlecloudsdk.api_lib.container.images import util
from googlecloudsdk.command_lib.container.binauthz import util as binauthz_util
from googlecloudsdk.core.exceptions import Error
import httplib2
class SigstoreAttestationImage(docker_image.DockerImage):
    """Creates a new image or appends a layers on top of an existing image.

  Adheres to the Sigstore Attestation spec:
  https://github.com/sigstore/cosign/blob/main/specs/ATTESTATION_SPEC.md.
  """

    def __init__(self, additional_blobs: List[bytes], base: Optional[docker_image.DockerImage]=None):
        """Creates a new Sigstore style image or extends a base image.

    Args:
      additional_blobs: additional attestations to be appended to the image.
      base: optional base DockerImage.
    """
        self._additional_blobs = collections.OrderedDict(((docker_digest.SHA256(blob), blob) for blob in additional_blobs))
        if base is not None:
            self._base = base
            self._base_manifest = json.loads(self._base.manifest())
            self._base_config_file = json.loads(self._base.config_file())
        else:
            self._base = None
            self._base_manifest = {'mediaType': docker_http.OCI_MANIFEST_MIME, 'schemaVersion': 2, 'config': {'digest': '', 'mediaType': docker_http.CONFIG_JSON_MIME, 'size': 0}, 'layers': []}
            self._base_config_file = dict()

    def add_layer(self, blob: bytes) -> None:
        self._additional_blobs[docker_digest.SHA256(blob)] = blob

    def config_file(self) -> Text:
        """Override."""
        config_file = self._base_config_file
        overrides = metadata.Overrides()
        overrides = overrides.Override(created_by=docker_name.USER_AGENT)
        layers = [_RemovePrefix(blob_sum, 'sha256:') for blob_sum in self._additional_blobs.keys()]
        overrides = overrides.Override(layers=layers)
        config_file = metadata.Override(config_file, options=overrides, architecture='', operating_system='')
        return json.dumps(config_file, sort_keys=True)

    def manifest(self) -> Text:
        """Override."""
        manifest = copy.deepcopy(self._base_manifest)
        for blob_sum, blob in self._additional_blobs.items():
            manifest['layers'].append({'digest': blob_sum, 'mediaType': DSSE_PAYLOAD_TYPE, 'size': len(blob), 'annotations': {'dev.cosignproject.cosign/signature': '', 'predicateType': BINAUTHZ_CUSTOM_PREDICATE}})
        config_file = self.config_file()
        utf8_encoded_config = config_file.encode('utf8')
        manifest['config']['digest'] = docker_digest.SHA256(utf8_encoded_config)
        manifest['config']['size'] = len(utf8_encoded_config)
        return json.dumps(manifest, sort_keys=True)

    def blob(self, digest: Text) -> bytes:
        """Override. Returns uncompressed blob."""
        if digest in self._additional_blobs:
            return self._additional_blobs[digest]
        if self._base:
            return self._base.blob(digest)
        raise Error('Digest not found: {}'.format(digest))

    def __enter__(self):
        """Override."""
        return self

    def __exit__(self, unused_type, unused_value, unused_traceback):
        """Override."""
        return