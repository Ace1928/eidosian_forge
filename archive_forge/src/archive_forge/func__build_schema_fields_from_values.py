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
def _build_schema_fields_from_values(peek, props):
    previous = OrderedDict()
    fill_missing = True
    fields = OrderedDict()
    for row in peek:
        _update_field_defs_from(props, row, fields, previous, fill_missing)
        fill_missing = False
    schema_fields = [item for item in fields.values()]
    return schema_fields