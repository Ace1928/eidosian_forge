from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import base64
import json
import os
import subprocess
from containerregistry.client import docker_name
def DetachSignatures(manifest):
    """Detach the signatures from the signed manifest and return the two halves.

  Args:
    manifest: a signed JSON manifest.
  Raises:
    BadManifestException: the provided manifest was improperly signed.
  Returns:
    a pair consisting of the manifest with the signature removed and a list of
    the removed signatures.
  """
    json_manifest = json.loads(manifest)
    signatures = json_manifest['signatures']
    if len(signatures) < 1:
        raise BadManifestException('Expected a signed manifest.')
    for sig in signatures:
        if 'protected' not in sig:
            raise BadManifestException('Signature is missing "protected" key')
    format_length, format_tail = _ExtractCommonProtectedRegion(signatures)
    suffix = _JoseBase64UrlDecode(format_tail)
    unsigned_manifest = manifest[0:format_length] + suffix
    return (unsigned_manifest, signatures)