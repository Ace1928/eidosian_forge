from functools import lru_cache
import os
import re
import shutil
import subprocess
import sys
import pytest
import pyarrow as pa
def test_record_batch(gdb_arrow):
    expected_prefix = 'arrow::RecordBatch with 2 columns, 3 rows'
    expected_suffix = '{["ints"] = arrow::ArrayData of type arrow::int32(), length 3, offset 0, null count 0 = {[0] = 1, [1] = 2, [2] = 3}, ["strs"] = arrow::ArrayData of type arrow::utf8(), length 3, offset 0, null count 1 = {[0] = "abc", [1] = null, [2] = "def"}}'
    expected = f'{expected_prefix} = {expected_suffix}'
    check_heap_repr(gdb_arrow, 'batch', expected)
    check_heap_repr(gdb_arrow, 'batch.get()', expected)
    expected = f'{expected_prefix}, 3 metadata items = {expected_suffix}'
    check_heap_repr(gdb_arrow, 'batch_with_metadata', expected)