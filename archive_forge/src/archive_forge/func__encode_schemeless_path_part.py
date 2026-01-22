import re
import sys
import string
import socket
from socket import AF_INET, AF_INET6
from typing import (
from unicodedata import normalize
from ._socket import inet_pton
from idna import encode as idna_encode, decode as idna_decode
def _encode_schemeless_path_part(text, maximal=True):
    """Percent-encode the first segment of a URL path for a URL without a
    scheme specified.
    """
    if maximal:
        bytestr = normalize('NFC', text).encode('utf8')
        return u''.join([_SCHEMELESS_PATH_PART_QUOTE_MAP[b] for b in bytestr])
    return u''.join([_SCHEMELESS_PATH_PART_QUOTE_MAP[t] if t in _SCHEMELESS_PATH_DELIMS else t for t in text])