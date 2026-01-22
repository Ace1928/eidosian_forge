import numpy as np
from .common import ut, TestCase
from h5py import Datatype
class TestCreation(TestCase):
    """
        Feature: repr() works sensibly on datatype objects
    """

    def test_repr(self):
        """ repr() on datatype objects """
        self.f['foo'] = np.dtype('S10')
        dt = self.f['foo']
        self.assertIsInstance(repr(dt), str)
        self.f.close()
        self.assertIsInstance(repr(dt), str)

    def test_appropriate_low_level_id(self):
        """ Binding a group to a non-TypeID identifier fails with ValueError """
        with self.assertRaises(ValueError):
            Datatype(self.f['/'].id)