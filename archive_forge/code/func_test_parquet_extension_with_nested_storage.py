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
def test_parquet_extension_with_nested_storage(tmpdir):
    import pyarrow.parquet as pq
    struct_array = pa.StructArray.from_arrays([pa.array([0, 1], type='int64'), pa.array([4, 5], type='int64')], names=['left', 'right'])
    list_array = pa.array([[1, 2, 3], [4, 5]], type=pa.list_(pa.int32()))
    mystruct_array = pa.ExtensionArray.from_storage(MyStructType(), struct_array)
    mylist_array = pa.ExtensionArray.from_storage(MyListType(list_array.type), list_array)
    orig_table = pa.table({'structs': mystruct_array, 'lists': mylist_array})
    filename = tmpdir / 'nested_extension_storage.parquet'
    pq.write_table(orig_table, filename)
    table = pq.read_table(filename)
    table.validate(full=True)
    assert table.column('structs').type == struct_array.type
    assert table.column('structs').combine_chunks() == struct_array
    assert table.column('lists').type == list_array.type
    assert table.column('lists').combine_chunks() == list_array
    with registered_extension_type(mystruct_array.type):
        with registered_extension_type(mylist_array.type):
            table = pq.read_table(filename)
            table.validate(full=True)
            assert table.column('structs').type == mystruct_array.type
            assert table.column('lists').type == mylist_array.type
            assert table == orig_table
            with pytest.raises(pa.ArrowInvalid, match='without all of its fields'):
                pq.ParquetFile(filename).read(columns=['structs.left'])