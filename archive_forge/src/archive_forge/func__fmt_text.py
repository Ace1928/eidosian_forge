from __future__ import division
import sys
import unicodedata
from functools import reduce
@classmethod
def _fmt_text(cls, x, **kw):
    """String formatting class-method."""
    return obj2unicode(x)