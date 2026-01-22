from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import subprocess
import tempfile
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.core import log
import six
def GetModulusExponentFromPublicKey(self, public_key, key_length=DEFAULT_KEY_LENGTH):
    """Returns a base64 encoded modulus and exponent from the public key."""
    key = StripKey(public_key)
    decoded_key = base64.b64decode(key)
    exponent = decoded_key[-3:]
    key_bytes = key_length // 8
    if key_length % 8:
        key_bytes += 1
    modulus_start = -5 - key_bytes
    modulus = decoded_key[modulus_start:-5]
    b64_mod = base64.b64encode(modulus)
    b64_exp = base64.b64encode(exponent)
    return (b64_mod, b64_exp)