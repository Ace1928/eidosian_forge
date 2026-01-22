from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import base64
import json
import os
import subprocess
from containerregistry.client import docker_name
def _AttachSignatures(manifest, signatures):
    """Attach the provided signatures to the provided naked manifest."""
    format_length, format_tail = _ExtractCommonProtectedRegion(signatures)
    prefix = manifest[0:format_length]
    suffix = _JoseBase64UrlDecode(format_tail)
    return '{prefix},"signatures":{signatures}{suffix}'.format(prefix=prefix, signatures=json.dumps(signatures, sort_keys=True), suffix=suffix)