import contextlib
import ctypes
import gc
import pyarrow as pa
import pytest
def test_roundtrip_batch_reader_capsule():
    batch = make_batch()
    capsule = batch.__arrow_c_stream__()
    assert PyCapsule_IsValid(capsule, b'arrow_array_stream') == 1
    imported_reader = pa.RecordBatchReader._import_from_c_capsule(capsule)
    assert imported_reader.schema == batch.schema
    assert imported_reader.read_next_batch().equals(batch)
    with pytest.raises(StopIteration):
        imported_reader.read_next_batch()