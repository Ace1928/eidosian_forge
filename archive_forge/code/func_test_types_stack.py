from functools import lru_cache
import os
import re
import shutil
import subprocess
import sys
import pytest
import pyarrow as pa
def test_types_stack(gdb_arrow):
    check_stack_repr(gdb_arrow, 'null_type', 'arrow::null()')
    check_stack_repr(gdb_arrow, 'bool_type', 'arrow::boolean()')
    check_stack_repr(gdb_arrow, 'date32_type', 'arrow::date32()')
    check_stack_repr(gdb_arrow, 'date64_type', 'arrow::date64()')
    check_stack_repr(gdb_arrow, 'time_type_s', 'arrow::time32(arrow::TimeUnit::SECOND)')
    check_stack_repr(gdb_arrow, 'time_type_ms', 'arrow::time32(arrow::TimeUnit::MILLI)')
    check_stack_repr(gdb_arrow, 'time_type_us', 'arrow::time64(arrow::TimeUnit::MICRO)')
    check_stack_repr(gdb_arrow, 'time_type_ns', 'arrow::time64(arrow::TimeUnit::NANO)')
    check_stack_repr(gdb_arrow, 'timestamp_type_s', 'arrow::timestamp(arrow::TimeUnit::SECOND)')
    check_stack_repr(gdb_arrow, 'timestamp_type_ms_timezone', 'arrow::timestamp(arrow::TimeUnit::MILLI, "Europe/Paris")')
    check_stack_repr(gdb_arrow, 'timestamp_type_us', 'arrow::timestamp(arrow::TimeUnit::MICRO)')
    check_stack_repr(gdb_arrow, 'timestamp_type_ns_timezone', 'arrow::timestamp(arrow::TimeUnit::NANO, "Europe/Paris")')
    check_stack_repr(gdb_arrow, 'day_time_interval_type', 'arrow::day_time_interval()')
    check_stack_repr(gdb_arrow, 'month_interval_type', 'arrow::month_interval()')
    check_stack_repr(gdb_arrow, 'month_day_nano_interval_type', 'arrow::month_day_nano_interval()')
    check_stack_repr(gdb_arrow, 'duration_type_s', 'arrow::duration(arrow::TimeUnit::SECOND)')
    check_stack_repr(gdb_arrow, 'duration_type_ns', 'arrow::duration(arrow::TimeUnit::NANO)')
    check_stack_repr(gdb_arrow, 'decimal128_type', 'arrow::decimal128(16, 5)')
    check_stack_repr(gdb_arrow, 'decimal256_type', 'arrow::decimal256(42, 12)')
    check_stack_repr(gdb_arrow, 'binary_type', 'arrow::binary()')
    check_stack_repr(gdb_arrow, 'string_type', 'arrow::utf8()')
    check_stack_repr(gdb_arrow, 'large_binary_type', 'arrow::large_binary()')
    check_stack_repr(gdb_arrow, 'large_string_type', 'arrow::large_utf8()')
    check_stack_repr(gdb_arrow, 'fixed_size_binary_type', 'arrow::fixed_size_binary(10)')
    check_stack_repr(gdb_arrow, 'list_type', 'arrow::list(arrow::uint8())')
    check_stack_repr(gdb_arrow, 'large_list_type', 'arrow::large_list(arrow::large_utf8())')
    check_stack_repr(gdb_arrow, 'fixed_size_list_type', 'arrow::fixed_size_list(arrow::float64(), 3)')
    check_stack_repr(gdb_arrow, 'map_type_unsorted', 'arrow::map(arrow::utf8(), arrow::binary(), keys_sorted=false)')
    check_stack_repr(gdb_arrow, 'map_type_sorted', 'arrow::map(arrow::utf8(), arrow::binary(), keys_sorted=true)')
    check_stack_repr(gdb_arrow, 'struct_type_empty', 'arrow::struct_({})')
    check_stack_repr(gdb_arrow, 'struct_type', 'arrow::struct_({arrow::field("ints", arrow::int8()), arrow::field("strs", arrow::utf8(), nullable=false)})')
    check_stack_repr(gdb_arrow, 'sparse_union_type', 'arrow::sparse_union(fields={arrow::field("ints", arrow::int8()), arrow::field("strs", arrow::utf8(), nullable=false)}, type_codes={7, 42})')
    check_stack_repr(gdb_arrow, 'dense_union_type', 'arrow::dense_union(fields={arrow::field("ints", arrow::int8()), arrow::field("strs", arrow::utf8(), nullable=false)}, type_codes={7, 42})')
    check_stack_repr(gdb_arrow, 'dict_type_unordered', 'arrow::dictionary(arrow::int16(), arrow::utf8(), ordered=false)')
    check_stack_repr(gdb_arrow, 'dict_type_ordered', 'arrow::dictionary(arrow::int16(), arrow::utf8(), ordered=true)')
    check_stack_repr(gdb_arrow, 'uuid_type', 'arrow::ExtensionType "extension<uuid>" with storage type arrow::fixed_size_binary(16)')