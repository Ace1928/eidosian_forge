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
def appendavro(table, target, schema=None, sample=9, **avro_args):
    """
    Append rows into a avro existing avro file or create a new one.

    The `target` argument can be either an existing avro file or the file 
    path for creating new one.

    The `schema` argument is checked against the schema of the existing file.
    So it must be the same schema as used by `toavro()` or the schema of the
    existing file.

    The `sample` argument (int, optional) defines how many rows are inspected
    for discovering the field types and building a schema for the avro file 
    when the `schema` argument is not passed.

    Additionally there are support for passing extra options in the 
    argument `**avro_args` that are fowarded directly to fastavro. Check the
    fastavro documentation for reference.

    See :meth:`petl.io.avro.toavro` method for more information and examples.

    .. versionadded:: 1.4.0

    """
    _write_toavro(table, target=target, mode='a+b', schema=schema, sample=sample, **avro_args)