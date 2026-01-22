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
def _get_error_details(target, num, err, record, schema):
    """show last row when failed writing for throubleshooting"""
    headers = _get_schema_header_names(schema)
    if isinstance(record, dict):
        table = [headers, list(record.values())]
    else:
        table = [headers, record]
    example = wrap(table).look()
    dest = ' output: %s' % target if isinstance(target, string_types) else ''
    printed = 'failed writing on row #%d: %s\n%s\n schema: %s\n%s'
    details = printed % (num, err, dest, schema, example)
    return details