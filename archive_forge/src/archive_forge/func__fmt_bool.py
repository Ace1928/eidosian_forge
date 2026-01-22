from __future__ import division
import sys
import unicodedata
from functools import reduce
@classmethod
def _fmt_bool(cls, x, **kw):
    """Boolean formatting class-method"""
    return str(bool(x))