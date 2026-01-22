from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import subprocess
import tempfile
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.core import log
import six
def StripKey(key):
    """Returns key with header, footer and all newlines removed."""
    key = key.strip()
    key_lines = key.split(b'\n')
    if not key_lines[0].startswith(b'-----') or not key_lines[-1].startswith(b'-----'):
        raise OpenSSLException('The following key does not appear to be in PEM format: \n{0}'.format(key))
    return b''.join(key_lines[1:-1])