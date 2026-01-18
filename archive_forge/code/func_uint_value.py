import io
import logging
import sys
import zlib
from typing import (
from . import settings
from .ascii85 import ascii85decode
from .ascii85 import asciihexdecode
from .ccitt import ccittfaxdecode
from .lzw import lzwdecode
from .psparser import LIT
from .psparser import PSException
from .psparser import PSObject
from .runlength import rldecode
from .utils import apply_png_predictor
def uint_value(x: object, n_bits: int) -> int:
    """Resolve number and interpret it as a two's-complement unsigned number"""
    xi = int_value(x)
    if xi > 0:
        return xi
    else:
        return xi + cast(int, 2 ** n_bits)