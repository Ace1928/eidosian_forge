import hypothesis as h
import pyarrow as pa
import pyarrow.tests.strategies as past
@h.given(past.all_schemas)
def test_schemas(schema):
    assert isinstance(schema, pa.lib.Schema)