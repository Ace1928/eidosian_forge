from __future__ import absolute_import
import re
from operator import itemgetter
import decimal
from .compat import binary_type, text_type, string_types, integer_types, PY3
from .decoder import PosInf
from .raw_json import RawJSON
def _encoder(o, _orig_encoder=_encoder, _encoding=self.encoding):
    if isinstance(o, binary_type):
        o = text_type(o, _encoding)
    return _orig_encoder(o)