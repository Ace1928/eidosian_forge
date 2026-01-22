import h5py
import numpy
import numpy.testing
import pytest
from .common import ut, TestCase
class TestReadDirectChunkToOut:

    def test_uncompressed_data(self, writable_file):
        ref_data = numpy.arange(16).reshape(4, 4)
        dataset = writable_file.create_dataset('uncompressed', data=ref_data, chunks=ref_data.shape)
        out = bytearray(ref_data.nbytes)
        filter_mask, chunk = dataset.id.read_direct_chunk((0, 0), out=out)
        assert numpy.array_equal(numpy.frombuffer(out, dtype=ref_data.dtype).reshape(ref_data.shape), ref_data)
        assert filter_mask == 0
        assert len(chunk) == ref_data.nbytes

    @pytest.mark.skipif(h5py.version.hdf5_version_tuple < (1, 10, 5), reason='chunk info requires HDF5 >= 1.10.5')
    @pytest.mark.skipif('gzip' not in h5py.filters.encode, reason='DEFLATE is not installed')
    def test_compressed_data(self, writable_file):
        ref_data = numpy.arange(16).reshape(4, 4)
        dataset = writable_file.create_dataset('gzip', data=ref_data, chunks=ref_data.shape, compression='gzip', compression_opts=9)
        chunk_info = dataset.id.get_chunk_info(0)
        out = bytearray(chunk_info.size)
        filter_mask, chunk = dataset.id.read_direct_chunk(chunk_info.chunk_offset, out=out)
        assert filter_mask == chunk_info.filter_mask
        assert len(chunk) == chunk_info.size
        assert out == dataset.id.read_direct_chunk(chunk_info.chunk_offset)[1]

    def test_fail_buffer_too_small(self, writable_file):
        ref_data = numpy.arange(16).reshape(4, 4)
        dataset = writable_file.create_dataset('uncompressed', data=ref_data, chunks=ref_data.shape)
        out = bytearray(ref_data.nbytes // 2)
        with pytest.raises(ValueError):
            dataset.id.read_direct_chunk((0, 0), out=out)

    def test_fail_buffer_readonly(self, writable_file):
        ref_data = numpy.arange(16).reshape(4, 4)
        dataset = writable_file.create_dataset('uncompressed', data=ref_data, chunks=ref_data.shape)
        out = bytes(ref_data.nbytes)
        with pytest.raises(BufferError):
            dataset.id.read_direct_chunk((0, 0), out=out)

    def test_fail_buffer_not_contiguous(self, writable_file):
        ref_data = numpy.arange(16).reshape(4, 4)
        dataset = writable_file.create_dataset('uncompressed', data=ref_data, chunks=ref_data.shape)
        array = numpy.empty(ref_data.shape + (2,), dtype=ref_data.dtype)
        out = array[:, :, ::2]
        with pytest.raises(ValueError):
            dataset.id.read_direct_chunk((0, 0), out=out)