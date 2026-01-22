from collections.abc import Sequence
from typing import Optional, Union, cast
from twisted.python.compat import nativeString
from twisted.web._responses import RESPONSES
class UnsupportedSpecialHeader(Exception):
    """
    A HTTP/2 request was received that contained a HTTP/2 pseudo-header field
    that is not recognised by Twisted.
    """