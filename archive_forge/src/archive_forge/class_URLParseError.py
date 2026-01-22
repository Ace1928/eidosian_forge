import re
import sys
import string
import socket
from socket import AF_INET, AF_INET6
from typing import (
from unicodedata import normalize
from ._socket import inet_pton
from idna import encode as idna_encode, decode as idna_decode
class URLParseError(ValueError):
    """Exception inheriting from :exc:`ValueError`, raised when failing to
    parse a URL. Mostly raised on invalid ports and IPv6 addresses.
    """
    pass