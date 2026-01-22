from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
from containerregistry.client.v2 import docker_image as v2_image
from containerregistry.client.v2 import util as v2_util
from containerregistry.client.v2_2 import docker_digest
from containerregistry.client.v2_2 import docker_http
from containerregistry.client.v2_2 import docker_image as v2_2_image
class V22FromV2(v2_2_image.DockerImage):
    """This compatibility interface serves the v2 interface from a v2_2 image."""

    def __init__(self, v2_img):
        """Constructor.

    Args:
      v2_img: a v2 DockerImage on which __enter__ has already been called.

    Raises:
      ValueError: an incorrectly typed argument was supplied.
    """
        self._v2_image = v2_img
        self._ProcessImage()

    def _ProcessImage(self):
        """Constructs schema 2 manifest from schema 1 manifest."""
        raw_manifest_schema1 = self._v2_image.manifest()
        manifest_schema1 = json.loads(raw_manifest_schema1)
        self._config_file = config_file([json.loads(history.get('v1Compatibility', '{}')) for history in reversed(manifest_schema1.get('history', []))], [self._GetDiffId(digest) for digest in reversed(self._v2_image.fs_layers())])
        config_bytes = self._config_file.encode('utf8')
        config_descriptor = {'mediaType': docker_http.CONFIG_JSON_MIME, 'size': len(config_bytes), 'digest': docker_digest.SHA256(config_bytes)}
        manifest_schema2 = {'schemaVersion': 2, 'mediaType': docker_http.MANIFEST_SCHEMA2_MIME, 'config': config_descriptor, 'layers': [{'mediaType': docker_http.LAYER_MIME, 'size': self._v2_image.blob_size(digest), 'digest': digest} for digest in reversed(self._v2_image.fs_layers())]}
        self._manifest = json.dumps(manifest_schema2, sort_keys=True)

    def _GetDiffId(self, digest):
        """Hash the uncompressed layer blob."""
        return docker_digest.SHA256(self._v2_image.uncompressed_blob(digest))

    def manifest(self):
        """Override."""
        return self._manifest

    def config_file(self):
        """Override."""
        return self._config_file

    def uncompressed_blob(self, digest):
        """Override."""
        return self._v2_image.uncompressed_blob(digest)

    def blob(self, digest):
        """Override."""
        return self._v2_image.blob(digest)

    def __enter__(self):
        return self

    def __exit__(self, unused_type, unused_value, unused_traceback):
        pass