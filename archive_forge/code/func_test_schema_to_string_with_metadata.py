from collections import OrderedDict
import sys
import weakref
import pytest
import numpy as np
import pyarrow as pa
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
def test_schema_to_string_with_metadata():
    lorem = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nulla accumsan vel\nturpis et mollis. Aliquam tincidunt arcu id tortor blandit blandit. Donec\neget leo quis lectus scelerisque varius. Class aptent taciti sociosqu ad\nlitora torquent per conubia nostra, per inceptos himenaeos. Praesent\nfaucibus, diam eu volutpat iaculis, tellus est porta ligula, a efficitur\nturpis nulla facilisis quam. Aliquam vitae lorem erat. Proin a dolor ac libero\ndignissim mollis vitae eu mauris. Quisque posuere tellus vitae massa\npellentesque sagittis. Aenean feugiat, diam ac dignissim fermentum, lorem\nsapien commodo massa, vel volutpat orci nisi eu justo. Nulla non blandit\nsapien. Quisque pretium vestibulum urna eu vehicula.'
    my_schema = pa.schema([pa.field('foo', 'int32', False, metadata={'key1': 'value1'}), pa.field('bar', 'string', True, metadata={'key3': 'value3'})], metadata={'lorem': lorem})
    assert my_schema.to_string() == "foo: int32 not null\n  -- field metadata --\n  key1: 'value1'\nbar: string\n  -- field metadata --\n  key3: 'value3'\n-- schema metadata --\nlorem: '" + lorem[:65] + "' + " + str(len(lorem) - 65)
    result = pa.schema([('f0', 'int32')], metadata={'key': 'value' + 'x' * 62}).to_string()
    assert result == "f0: int32\n-- schema metadata --\nkey: 'valuexxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'"
    assert my_schema.to_string(truncate_metadata=False) == "foo: int32 not null\n  -- field metadata --\n  key1: 'value1'\nbar: string\n  -- field metadata --\n  key3: 'value3'\n-- schema metadata --\nlorem: '{}'".format(lorem)
    assert my_schema.to_string(truncate_metadata=False, show_field_metadata=False) == "foo: int32 not null\nbar: string\n-- schema metadata --\nlorem: '{}'".format(lorem)
    assert my_schema.to_string(truncate_metadata=False, show_schema_metadata=False) == "foo: int32 not null\n  -- field metadata --\n  key1: 'value1'\nbar: string\n  -- field metadata --\n  key3: 'value3'"
    assert my_schema.to_string(truncate_metadata=False, show_field_metadata=False, show_schema_metadata=False) == 'foo: int32 not null\nbar: string'