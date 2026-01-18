from functools import lru_cache
import os
import re
import shutil
import subprocess
import sys
import pytest
import pyarrow as pa
def test_types_heap(gdb_arrow):
    check_heap_repr(gdb_arrow, 'heap_null_type', 'arrow::null()')
    check_heap_repr(gdb_arrow, 'heap_bool_type', 'arrow::boolean()')
    check_heap_repr(gdb_arrow, 'heap_time_type_ns', 'arrow::time64(arrow::TimeUnit::NANO)')
    check_heap_repr(gdb_arrow, 'heap_timestamp_type_ns_timezone', 'arrow::timestamp(arrow::TimeUnit::NANO, "Europe/Paris")')
    check_heap_repr(gdb_arrow, 'heap_decimal128_type', 'arrow::decimal128(16, 5)')
    check_heap_repr(gdb_arrow, 'heap_list_type', 'arrow::list(arrow::uint8())')
    check_heap_repr(gdb_arrow, 'heap_large_list_type', 'arrow::large_list(arrow::large_utf8())')
    check_heap_repr(gdb_arrow, 'heap_fixed_size_list_type', 'arrow::fixed_size_list(arrow::float64(), 3)')
    check_heap_repr(gdb_arrow, 'heap_map_type', 'arrow::map(arrow::utf8(), arrow::binary(), keys_sorted=false)')
    check_heap_repr(gdb_arrow, 'heap_struct_type', 'arrow::struct_({arrow::field("ints", arrow::int8()), arrow::field("strs", arrow::utf8(), nullable=false)})')
    check_heap_repr(gdb_arrow, 'heap_dict_type', 'arrow::dictionary(arrow::int16(), arrow::utf8(), ordered=false)')
    check_heap_repr(gdb_arrow, 'heap_uuid_type', 'arrow::ExtensionType "extension<uuid>" with storage type arrow::fixed_size_binary(16)')