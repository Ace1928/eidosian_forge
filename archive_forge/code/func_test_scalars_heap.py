from functools import lru_cache
import os
import re
import shutil
import subprocess
import sys
import pytest
import pyarrow as pa
def test_scalars_heap(gdb_arrow):
    check_heap_repr(gdb_arrow, 'heap_null_scalar', 'arrow::NullScalar')
    check_heap_repr(gdb_arrow, 'heap_bool_scalar', 'arrow::BooleanScalar of value true')
    check_heap_repr(gdb_arrow, 'heap_decimal128_scalar', 'arrow::Decimal128Scalar of value 123.4567 [precision=10, scale=4]')
    check_heap_repr(gdb_arrow, 'heap_decimal256_scalar', 'arrow::Decimal256Scalar of value 123456789012345678901234567890123456789012.3456 [precision=50, scale=4]')
    check_heap_repr(gdb_arrow, 'heap_map_scalar', 'arrow::MapScalar of type arrow::map(arrow::utf8(), arrow::int32(), keys_sorted=false), value length 2, offset 0, null count 0')
    check_heap_repr(gdb_arrow, 'heap_map_scalar_null', 'arrow::MapScalar of type arrow::map(arrow::utf8(), arrow::int32(), keys_sorted=false), null value')