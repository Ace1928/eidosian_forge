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
def _get_definition_from_array(prop, val, prev):
    afield = None
    for item in iter(val):
        if item is None:
            continue
        field2, curr2 = _get_definition_from_type_of(prop, item, prev)
        if field2 is not None:
            afield = field2
        if curr2 is not None:
            prev = curr2
    bfield = 'string' if afield is None else afield
    tdef = {'type': 'array', 'items': bfield}
    return (tdef, prev)