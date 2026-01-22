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
def _parse_dn_component(name, sep=b',', decode_remainder=True):
    m = DN_COMPONENT_START_RE.match(name)
    if not m:
        raise OpenSSLObjectError(u'cannot start part in "{0}"'.format(to_text(name)))
    oid = cryptography_name_to_oid(to_text(m.group(1)))
    idx = len(m.group(0))
    decoded_name = []
    sep_str = sep + b'\\'
    if decode_remainder:
        length = len(name)
        if length > idx and name[idx:idx + 1] == b'#':
            idx += 1
            while idx + 1 < length:
                ch1 = name[idx:idx + 1]
                ch2 = name[idx + 1:idx + 2]
                idx1 = DN_HEX_LETTER.find(ch1.lower())
                idx2 = DN_HEX_LETTER.find(ch2.lower())
                if idx1 < 0 or idx2 < 0:
                    raise OpenSSLObjectError(u'Invalid hex sequence entry "{0}"'.format(to_text(ch1 + ch2)))
                idx += 2
                decoded_name.append(_int_to_byte(idx1 * 16 + idx2))
        else:
            while idx < length:
                i = idx
                while i < length and name[i:i + 1] not in sep_str:
                    i += 1
                if i > idx:
                    decoded_name.append(name[idx:i])
                    idx = i
                while idx + 1 < length and name[idx:idx + 1] == b'\\':
                    ch = name[idx + 1:idx + 2]
                    idx1 = DN_HEX_LETTER.find(ch.lower())
                    if idx1 >= 0:
                        if idx + 2 >= length:
                            raise OpenSSLObjectError(u'Hex escape sequence "\\{0}" incomplete at end of string'.format(to_text(ch)))
                        ch2 = name[idx + 2:idx + 3]
                        idx2 = DN_HEX_LETTER.find(ch2.lower())
                        if idx2 < 0:
                            raise OpenSSLObjectError(u'Hex escape sequence "\\{0}" has invalid second letter'.format(to_text(ch + ch2)))
                        ch = _int_to_byte(idx1 * 16 + idx2)
                        idx += 1
                    idx += 2
                    decoded_name.append(ch)
                if idx < length and name[idx:idx + 1] == sep:
                    break
    else:
        decoded_name.append(name[idx:])
        idx = len(name)
    return (x509.NameAttribute(oid, to_text(b''.join(decoded_name))), name[idx:])