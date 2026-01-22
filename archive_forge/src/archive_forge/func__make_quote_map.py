import re
import sys
import string
import socket
from socket import AF_INET, AF_INET6
from typing import (
from unicodedata import normalize
from ._socket import inet_pton
from idna import encode as idna_encode, decode as idna_decode
def _make_quote_map(safe_chars):
    ret = {}
    for i, v in zip(range(256), range(256)):
        c = chr(v)
        if c in safe_chars:
            ret[c] = ret[v] = c
        else:
            ret[c] = ret[v] = '%{0:02X}'.format(i)
    return ret