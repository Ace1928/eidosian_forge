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
def _fix_missing_headers(table, schema):
    """add missing columns headers from schema"""
    if schema is None or 'fields' not in schema:
        return table
    sample, table2 = iterpeek(table, 2)
    cols = fieldnames(sample)
    headers = _get_schema_header_names(schema)
    if len(cols) >= len(headers):
        return table2
    table3 = setheader(table2, headers)
    return table3