import contextlib
import os
import shutil
import subprocess
import weakref
from uuid import uuid4, UUID
import sys
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
import pytest
@pytest.mark.parquet
def test_parquet_extension_nested_in_extension(tmpdir):
    import pyarrow.parquet as pq
    inner_ext_type = IntegerType()
    inner_storage = pa.array([4, 5, 6, 7], type=pa.int64())
    inner_ext_array = pa.ExtensionArray.from_storage(inner_ext_type, inner_storage)
    list_array = pa.ListArray.from_arrays([0, 1, None, 3], inner_ext_array)
    mylist_array = pa.ExtensionArray.from_storage(MyListType(list_array.type), list_array)
    orig_table = pa.table({'lists': mylist_array})
    filename = tmpdir / 'ext_of_list_of_ext.parquet'
    pq.write_table(orig_table, filename)
    table = pq.read_table(filename)
    assert table.column(0).type == pa.list_(pa.int64())
    with registered_extension_type(mylist_array.type):
        with registered_extension_type(inner_ext_array.type):
            table = pq.read_table(filename)
            assert table.column(0).type == mylist_array.type
            assert table == orig_table