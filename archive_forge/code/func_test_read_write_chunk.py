import h5py
import numpy
import numpy.testing
import pytest
from .common import ut, TestCase
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