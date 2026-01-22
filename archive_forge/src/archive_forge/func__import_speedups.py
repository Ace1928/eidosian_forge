from __future__ import absolute_import
import re
from operator import itemgetter
import decimal
from .compat import binary_type, text_type, string_types, integer_types, PY3
from .decoder import PosInf
from .raw_json import RawJSON
def _import_speedups():
    try:
        from . import _speedups
        return (_speedups.encode_basestring_ascii, _speedups.make_encoder)
    except ImportError:
        return (None, None)