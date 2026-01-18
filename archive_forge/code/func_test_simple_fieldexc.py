import numpy as np
import h5py
import h5py._hl.selections as sel
import h5py._hl.selections2 as sel2
from .common import TestCase, ut
def test_simple_fieldexc(self):
    """ Field names for non-field types raises ValueError """
    dt = np.dtype('i')
    with self.assertRaises(ValueError):
        out, format = sel2.read_dtypes(dt, ('a',))