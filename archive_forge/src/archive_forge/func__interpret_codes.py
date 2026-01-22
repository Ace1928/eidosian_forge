from __future__ import absolute_import
import re
import sys
import os
import tokenize
from io import StringIO
from ._looper import looper
from .compat3 import bytes, unicode_, basestring_, next, is_unicode, coerce_text
def _interpret_codes(self, codes, ns, out, defs):
    __traceback_hide__ = True
    for item in codes:
        if isinstance(item, basestring_):
            out.append(item)
        else:
            self._interpret_code(item, ns, out, defs)