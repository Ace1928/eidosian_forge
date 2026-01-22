import re
import sys
import string
import socket
from socket import AF_INET, AF_INET6
from typing import (
from unicodedata import normalize
from ._socket import inet_pton
from idna import encode as idna_encode, decode as idna_decode
def _decode_path_part(text, normalize_case=False, encode_stray_percents=False):
    """
    >>> _decode_path_part(u'%61%77%2f%7a')
    u'aw%2fz'
    >>> _decode_path_part(u'%61%77%2f%7a', normalize_case=True)
    u'aw%2Fz'
    """
    return _percent_decode(text, normalize_case=normalize_case, encode_stray_percents=encode_stray_percents, _decode_map=_PATH_DECODE_MAP)