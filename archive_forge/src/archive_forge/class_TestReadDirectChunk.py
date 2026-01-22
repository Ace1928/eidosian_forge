import h5py
import numpy
import numpy.testing
import pytest
from .common import ut, TestCase
@ut.skipIf('gzip' not in h5py.filters.encode, 'DEFLATE is not installed')
class TestReadDirectChunk(TestCase):

    def test_read_compressed_offsets(self):
        filename = self.mktemp().encode()
        with h5py.File(filename, 'w') as filehandle:
            frame = numpy.arange(16).reshape(4, 4)
            frame_dataset = filehandle.create_dataset('frame', data=frame, compression='gzip', compression_opts=9)
            dataset = filehandle.create_dataset('compressed_chunked', data=[frame, frame, frame], compression='gzip', compression_opts=9, chunks=(1,) + frame.shape)
            filter_mask, compressed_frame = frame_dataset.id.read_direct_chunk((0, 0))
            self.assertEqual(filter_mask, 0)
            for i in range(dataset.shape[0]):
                filter_mask, data = dataset.id.read_direct_chunk((i, 0, 0))
                self.assertEqual(compressed_frame, data)
                self.assertEqual(filter_mask, 0)

    def test_read_uncompressed_offsets(self):
        filename = self.mktemp().encode()
        frame = numpy.arange(16).reshape(4, 4)
        with h5py.File(filename, 'w') as filehandle:
            dataset = filehandle.create_dataset('frame', maxshape=(1,) + frame.shape, shape=(1,) + frame.shape, compression='gzip', compression_opts=9)
            DISABLE_ALL_FILTERS = 4294967295
            dataset.id.write_direct_chunk((0, 0, 0), frame.tobytes(), filter_mask=DISABLE_ALL_FILTERS)
        with h5py.File(filename, 'r') as filehandle:
            dataset = filehandle['frame']
            filter_mask, compressed_frame = dataset.id.read_direct_chunk((0, 0, 0))
        self.assertNotEqual(filter_mask, 0)
        self.assertEqual(compressed_frame, frame.tobytes())

    def test_read_write_chunk(self):
        filename = self.mktemp().encode()
        with h5py.File(filename, 'w') as filehandle:
            frame = numpy.arange(16).reshape(4, 4)
            frame_dataset = filehandle.create_dataset('source', data=frame, compression='gzip', compression_opts=9)
            filter_mask, compressed_frame = frame_dataset.id.read_direct_chunk((0, 0))
            dataset = filehandle.create_dataset('created', shape=frame_dataset.shape, maxshape=frame_dataset.shape, chunks=frame_dataset.chunks, dtype=frame_dataset.dtype, compression='gzip', compression_opts=9)
            dataset.id.write_direct_chunk((0, 0), compressed_frame, filter_mask=filter_mask)
        with h5py.File(filename, 'r') as filehandle:
            dataset = filehandle['created'][...]
            numpy.testing.assert_array_equal(dataset, frame)