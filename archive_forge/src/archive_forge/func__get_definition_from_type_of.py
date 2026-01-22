from __future__ import absolute_import, division, print_function
import sys
import math
from collections import OrderedDict
from datetime import datetime, date, time
from decimal import Decimal
from petl.compat import izip_longest, text_type, string_types, PY3
from petl.io.sources import read_source_from_arg, write_source_from_arg
from petl.transform.headers import skip, setheader
from petl.util.base import Table, dicts, fieldnames, iterpeek, wrap
def _get_definition_from_type_of(prop, val, prev):
    tdef = None
    curr = None
    if isinstance(val, datetime):
        tdef = {'type': 'long', 'logicalType': 'timestamp-millis'}
    elif isinstance(val, time):
        tdef = {'type': 'int', 'logicalType': 'time-millis'}
    elif isinstance(val, date):
        tdef = {'type': 'int', 'logicalType': 'date'}
    elif isinstance(val, Decimal):
        curr, precision, scale = _get_precision_from_decimal(curr, val, prev)
        tdef = {'type': 'bytes', 'logicalType': 'decimal', 'precision': precision, 'scale': scale}
    elif isinstance(val, bytes):
        tdef = 'bytes'
    elif isinstance(val, list):
        tdef, curr = _get_definition_from_array(prop, val, prev)
    elif isinstance(val, bool):
        tdef = 'boolean'
    elif isinstance(val, float):
        tdef = 'double'
    elif isinstance(val, int):
        tdef = 'long'
    elif val is not None:
        tdef = 'string'
    else:
        return (None, None)
    return (tdef, curr)