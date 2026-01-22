from __future__ import absolute_import, division, print_function
import base64
import binascii
import re
import sys
import traceback
from ansible.module_utils.common.text.converters import to_text, to_bytes, to_native
from ansible.module_utils.six.moves.urllib.parse import urlparse, urlunparse, ParseResult
from ._asn1 import serialize_asn1_string_as_der
from ansible_collections.community.crypto.plugins.module_utils.version import LooseVersion
from ansible.module_utils.basic import missing_required_lib
from .basic import (
from ._objects import (
from ._obj2txt import obj2txt
def _compare_private_keys(key1, key2, clazz, has_no_private_bytes=False):
    a = isinstance(key1, clazz)
    b = isinstance(key2, clazz)
    if not (a or b):
        return None
    if not a or not b:
        return False
    if has_no_private_bytes:
        return cryptography_compare_public_keys(a.public_key(), b.public_key())
    encryption_algorithm = cryptography.hazmat.primitives.serialization.NoEncryption()
    a = key1.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, encryption_algorithm=encryption_algorithm)
    b = key2.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, encryption_algorithm=encryption_algorithm)
    return a == b