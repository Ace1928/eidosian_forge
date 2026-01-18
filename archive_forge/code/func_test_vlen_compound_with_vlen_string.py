import numpy as np
from .common import TestCase, ut
import h5py
from h5py import h5a, h5s, h5t
from h5py import File
from h5py._hl.base import is_empty_dataspace
def test_vlen_compound_with_vlen_string(self):
    """ Compound scalars with vlen compounds containing vlen strings can be written and read """
    dt_inner = np.dtype([('a', h5py.string_dtype()), ('b', h5py.string_dtype())])
    dt = np.dtype([('f', h5py.vlen_dtype(dt_inner))])
    data = np.array((np.array([(b'apples', b'bananas'), (b'peaches', b'oranges')], dtype=dt_inner),), dtype=dt)[()]
    self.f.attrs['x'] = data
    out = self.f.attrs['x']
    self.assertArrayEqual(out, data, check_alignment=False)