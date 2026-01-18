import binascii
import codecs
import hashlib
import re
from binascii import unhexlify
from math import log10
from typing import Any, Callable, ClassVar, Dict, Optional, Sequence, Union, cast
from .._codecs import _pdfdoc_encoding_rev
from .._protocols import PdfObjectProtocol, PdfWriterProtocol
from .._utils import (
from ..errors import STREAM_TRUNCATED_PREMATURELY, PdfReadError, PdfStreamError
@staticmethod
def unnumber(sin: bytes) -> bytes:
    i = sin.find(b'#', 0)
    while i >= 0:
        try:
            sin = sin[:i] + unhexlify(sin[i + 1:i + 3]) + sin[i + 3:]
            i = sin.find(b'#', i + 1)
        except ValueError:
            i = i + 1
    return sin