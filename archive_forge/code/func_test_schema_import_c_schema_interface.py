from collections import OrderedDict
from collections.abc import Iterator
from functools import partial
import datetime
import sys
import pytest
import hypothesis as h
import hypothesis.strategies as st
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.types as types
import pyarrow.tests.strategies as past
def test_schema_import_c_schema_interface():

    class Wrapper:

        def __init__(self, schema):
            self.schema = schema

        def __arrow_c_schema__(self):
            return self.schema.__arrow_c_schema__()
    schema = pa.schema([pa.field('field_name', pa.int32())])
    wrapped_schema = Wrapper(schema)
    assert pa.schema(wrapped_schema) == schema