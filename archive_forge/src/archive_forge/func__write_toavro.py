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
def _write_toavro(table, target, mode, schema, sample, codec='deflate', compression_level=None, **avro_args):
    if table is None:
        return
    if not schema:
        schema, table2 = _build_schema_from_values(table, sample)
    else:
        table2 = _fix_missing_headers(table, schema)
    rows = dicts(table2) if PY3 else _ordered_dict_iterator(table2)
    target2 = write_source_from_arg(target, mode=mode)
    with target2.open(mode) as target_file:
        from fastavro import parse_schema
        from fastavro.write import Writer
        parsed_schema = parse_schema(schema)
        writer = Writer(fo=target_file, schema=parsed_schema, codec=codec, compression_level=compression_level, **avro_args)
        num = 1
        for record in rows:
            try:
                writer.write(record)
                num = num + 1
            except ValueError as verr:
                vmsg = _get_error_details(target, num, verr, record, schema)
                _raise_error(ValueError, vmsg)
            except TypeError as terr:
                tmsg = _get_error_details(target, num, terr, record, schema)
                _raise_error(TypeError, tmsg)
        writer.flush()