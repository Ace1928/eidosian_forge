import io
import numpy as np
import pyarrow as pa
from pyarrow.tests import util
def _roundtrip_table(table, read_table_kwargs=None, write_table_kwargs=None):
    read_table_kwargs = read_table_kwargs or {}
    write_table_kwargs = write_table_kwargs or {}
    writer = pa.BufferOutputStream()
    _write_table(table, writer, **write_table_kwargs)
    reader = pa.BufferReader(writer.getvalue())
    return _read_table(reader, **read_table_kwargs)