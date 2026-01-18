import tempfile
import shutil
import os
import numpy as np
from h5py import File, special_dtype
from h5py._hl.files import direct_vfd
from .common import ut, TestCase
from .common import ut, TestCase, UNICODE_FILENAMES, closed_tempfile
def test_vlen_strings(self):
    dn_tmp = tempfile.mkdtemp('h5py.lowtest.test_h5f.TestVlenStrings.test_vlen_strings')
    fn_h5 = os.path.join(dn_tmp, 'test.h5')
    try:
        with File(fn_h5, mode='w') as h:
            vlen_str = special_dtype(vlen=str)
            vlen_vlen_str = special_dtype(vlen=vlen_str)
            ds = h.create_dataset('/com', (2,), dtype=vlen_vlen_str)
            ds[0] = np.array(['a', 'b', 'c'], dtype=vlen_vlen_str)
            ds[1] = np.array(['d', 'e', 'f', 'g'], dtype=vlen_vlen_str)
        with File(fn_h5, 'r') as h:
            ds = h['com']
            assert ds[0].tolist() == [b'a', b'b', b'c']
            assert ds[1].tolist() == [b'd', b'e', b'f', b'g']
    finally:
        shutil.rmtree(dn_tmp)