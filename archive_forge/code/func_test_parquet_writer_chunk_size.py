import pytest
import pyarrow as pa
from pyarrow import fs
from pyarrow.filesystem import FileSystem, LocalFileSystem
@pytest.mark.large_memory
@pytest.mark.pandas
def test_parquet_writer_chunk_size(tempdir):
    default_chunk_size = 1024 * 1024
    abs_max_chunk_size = 64 * 1024 * 1024

    def check_chunk_size(data_size, chunk_size, expect_num_chunks):
        table = pa.Table.from_arrays([_range_integers(data_size, 'b')], names=['x'])
        if chunk_size is None:
            pq.write_table(table, tempdir / 'test.parquet')
        else:
            pq.write_table(table, tempdir / 'test.parquet', row_group_size=chunk_size)
        metadata = pq.read_metadata(tempdir / 'test.parquet')
        expected_chunk_size = default_chunk_size if chunk_size is None else chunk_size
        assert metadata.num_row_groups == expect_num_chunks
        latched_chunk_size = min(expected_chunk_size, abs_max_chunk_size)
        for chunk_idx in range(expect_num_chunks - 1):
            assert metadata.row_group(chunk_idx).num_rows == latched_chunk_size
        remainder = data_size - expected_chunk_size * (expect_num_chunks - 1)
        if remainder == 0:
            assert metadata.row_group(expect_num_chunks - 1).num_rows == latched_chunk_size
        else:
            assert metadata.row_group(expect_num_chunks - 1).num_rows == remainder
    check_chunk_size(default_chunk_size * 2, default_chunk_size - 100, 3)
    check_chunk_size(default_chunk_size * 2, default_chunk_size, 2)
    check_chunk_size(default_chunk_size * 2, default_chunk_size + 100, 2)
    check_chunk_size(default_chunk_size + 100, default_chunk_size + 100, 1)
    check_chunk_size(abs_max_chunk_size * 2, abs_max_chunk_size * 2, 2)
    check_chunk_size(default_chunk_size, None, 1)
    check_chunk_size(default_chunk_size + 1, None, 2)