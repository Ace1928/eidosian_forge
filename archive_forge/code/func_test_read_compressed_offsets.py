import h5py
import numpy
import numpy.testing
import pytest
from .common import ut, TestCase
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