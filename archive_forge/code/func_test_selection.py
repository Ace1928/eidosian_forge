import numpy as np
import h5py
import h5py._hl.selections as sel
import h5py._hl.selections2 as sel2
from .common import TestCase, ut
def test_selection(self):
    dset = self.f.create_dataset('dset', (100, 100))
    regref = dset.regionref[0:100, 0:100]
    st = sel.select((10,), list([1, 2, 3]), dset)
    self.assertIsInstance(st, sel.FancySelection)
    st = sel.select((10,), ((1, 2, 3),), dset)
    self.assertIsInstance(st, sel.FancySelection)
    st1 = sel.select((5,), np.array([True, False, False, False, True]), dset)
    self.assertIsInstance(st1, sel.PointSelection)
    st2 = sel.select((10,), 1, dset)
    self.assertIsInstance(st2, sel.SimpleSelection)
    with self.assertRaises(TypeError):
        sel.select((100,), 'foo', dset)
    st3 = sel.select((100, 100), regref, dset)
    self.assertIsInstance(st3, sel.Selection)
    with self.assertRaises(TypeError):
        sel.select((100,), regref, None)
    with self.assertRaises(TypeError):
        sel.select((100,), regref, dset)
    st4 = sel.select((100, 100), st3, dset)
    self.assertEqual(st4, st3)
    with self.assertRaises(TypeError):
        sel.select((100,), st3, dset)