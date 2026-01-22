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
def _build_schema_from_values(table, sample):
    samples, table2 = iterpeek(table, sample + 1)
    props = fieldnames(samples)
    peek = skip(samples, 1)
    schema_fields = _build_schema_fields_from_values(peek, props)
    schema_source = _build_schema_with(schema_fields)
    return (schema_source, table2)