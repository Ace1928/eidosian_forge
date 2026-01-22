import contextlib
import ctypes
import gc
import pyarrow as pa
import pytest
def _export_import_batch_reader(ptr_stream, reader_factory):
    batches = make_batches()
    schema = batches[0].schema
    reader = reader_factory(schema, batches)
    reader._export_to_c(ptr_stream)
    del reader, batches
    reader_new = pa.RecordBatchReader._import_from_c(ptr_stream)
    assert reader_new.schema == schema
    got_batches = list(reader_new)
    del reader_new
    assert got_batches == make_batches()
    if pd is not None:
        batches = make_batches()
        schema = batches[0].schema
        expected_df = pa.Table.from_batches(batches).to_pandas()
        reader = reader_factory(schema, batches)
        reader._export_to_c(ptr_stream)
        del reader, batches
        reader_new = pa.RecordBatchReader._import_from_c(ptr_stream)
        got_df = reader_new.read_pandas()
        del reader_new
        tm.assert_frame_equal(expected_df, got_df)