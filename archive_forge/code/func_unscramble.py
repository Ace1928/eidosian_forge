from __future__ import (absolute_import, division, print_function)
import base64
import random
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.module_utils.six import PY2
def unscramble(value, key):
    """Do NOT use this for cryptographic purposes!"""
    if len(key) < 1:
        raise ValueError('Key must be at least one byte')
    if not value.startswith(u'=S='):
        raise ValueError('Value does not start with indicator')
    value = base64.b64decode(value[3:])
    if PY2:
        k = ord(key[0])
        value = b''.join([chr(k ^ ord(b)) for b in value])
    else:
        k = key[0]
        value = bytes([k ^ b for b in value])
    return to_text(value)