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
def _parse_hex(bytesstr):
    if bytesstr is None:
        return bytesstr
    data = ''.join(['0' * (2 - len(p)) + p if len(p) < 2 else p for p in to_text(bytesstr).split(':')])
    data = binascii.unhexlify(data)
    return data