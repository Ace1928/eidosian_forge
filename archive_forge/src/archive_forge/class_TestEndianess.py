import pytest
import numpy as np
from h5py import File
from .common import TestCase
from .data_files import get_data_file_path
class TestEndianess(TestCase):

    def test_simple_int_be(self):
        fname = self.mktemp()
        arr = np.ndarray(shape=(1,), dtype='>i4', buffer=bytearray([0, 1, 3, 2]))
        be_number = 0 * 256 ** 3 + 1 * 256 ** 2 + 3 * 256 ** 1 + 2 * 256 ** 0
        with File(fname, mode='w') as f:
            f.create_dataset('int', data=arr)
        with File(fname, mode='r') as f:
            assert f['int'][()][0] == be_number