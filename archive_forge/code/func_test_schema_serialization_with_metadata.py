from collections import UserList
import io
import pathlib
import pytest
import socket
import threading
import weakref
import numpy as np
import pyarrow as pa
from pyarrow.tests.util import changed_environ, invoke_script
def test_schema_serialization_with_metadata():
    field_metadata = {b'foo': b'bar', b'kind': b'field'}
    schema_metadata = {b'foo': b'bar', b'kind': b'schema'}
    f0 = pa.field('a', pa.int8())
    f1 = pa.field('b', pa.string(), metadata=field_metadata)
    schema = pa.schema([f0, f1], metadata=schema_metadata)
    s_schema = schema.serialize()
    recons_schema = pa.ipc.read_schema(s_schema)
    assert recons_schema.equals(schema)
    assert recons_schema.metadata == schema_metadata
    assert recons_schema[0].metadata is None
    assert recons_schema[1].metadata == field_metadata