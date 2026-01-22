import re
import sys
import string
import socket
from socket import AF_INET, AF_INET6
from typing import (
from unicodedata import normalize
from ._socket import inet_pton
from idna import encode as idna_encode, decode as idna_decode
def _encode_query_key(text, maximal=True):
    """
    Percent-encode a single query string key or value.
    """
    if maximal:
        bytestr = normalize('NFC', text).encode('utf8')
        return u''.join([_QUERY_KEY_QUOTE_MAP[b] for b in bytestr])
    return u''.join([_QUERY_KEY_QUOTE_MAP[t] if t in _QUERY_KEY_DELIMS else t for t in text])