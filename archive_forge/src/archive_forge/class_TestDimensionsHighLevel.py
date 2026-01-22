import sys
import numpy as np
from .common import ut, TestCase
from h5py import File, Group, Dataset
import h5py
class TestDimensionsHighLevel(BaseDataset):

    def test_len(self):
        self.assertEqual(len(self.f['data'].dims[0]), 0)
        self.assertEqual(len(self.f['data'].dims[1]), 1)
        self.assertEqual(len(self.f['data'].dims[2]), 2)
        self.assertEqual(len(self.f['data2'].dims[0]), 0)
        self.assertEqual(len(self.f['data2'].dims[1]), 0)
        self.assertEqual(len(self.f['data2'].dims[2]), 0)

    def test_get_label(self):
        self.assertEqual(self.f['data'].dims[2].label, 'x')
        self.assertEqual(self.f['data'].dims[1].label, '')
        self.assertEqual(self.f['data'].dims[0].label, 'z')
        self.assertEqual(self.f['data2'].dims[2].label, '')
        self.assertEqual(self.f['data2'].dims[1].label, '')
        self.assertEqual(self.f['data2'].dims[0].label, '')

    def test_set_label(self):
        self.f['data'].dims[0].label = 'foo'
        self.assertEqual(self.f['data'].dims[2].label, 'x')
        self.assertEqual(self.f['data'].dims[1].label, '')
        self.assertEqual(self.f['data'].dims[0].label, 'foo')

    def test_detach_scale(self):
        self.f['data'].dims[2].detach_scale(self.f['x1'])
        self.assertEqual(len(self.f['data'].dims[2]), 1)
        self.assertEqual(self.f['data'].dims[2][0], self.f['x2'])
        self.f['data'].dims[2].detach_scale(self.f['x2'])
        self.assertEqual(len(self.f['data'].dims[2]), 0)

    def test_attach_scale(self):
        self.f['x3'] = self.f['x2'][...]
        self.f['data'].dims[2].attach_scale(self.f['x3'])
        self.assertEqual(len(self.f['data'].dims[2]), 3)
        self.assertEqual(self.f['data'].dims[2][2], self.f['x3'])

    def test_get_dimension_scale(self):
        self.assertEqual(self.f['data'].dims[2][0], self.f['x1'])
        with self.assertRaises(RuntimeError):
            (self.f['data2'].dims[2][0], self.f['x2'])
        self.assertEqual(self.f['data'].dims[2][''], self.f['x1'])
        self.assertEqual(self.f['data'].dims[2]['x2 name'], self.f['x2'])

    def test_get_items(self):
        self.assertEqual(self.f['data'].dims[2].items(), [('', self.f['x1']), ('x2 name', self.f['x2'])])

    def test_get_keys(self):
        self.assertEqual(self.f['data'].dims[2].keys(), ['', 'x2 name'])

    def test_get_values(self):
        self.assertEqual(self.f['data'].dims[2].values(), [self.f['x1'], self.f['x2']])

    def test_iter(self):
        self.assertEqual([i for i in self.f['data'].dims[2]], ['', 'x2 name'])

    def test_repr(self):
        ds = self.f['data']
        self.assertEqual(repr(ds.dims[2])[1:16], '"x" dimension 2')
        self.f.close()
        self.assertIsInstance(repr(ds.dims), str)

    def test_attributes(self):
        self.f['data2'].attrs['DIMENSION_LIST'] = self.f['data'].attrs['DIMENSION_LIST']
        self.assertEqual(len(self.f['data2'].dims[0]), 0)
        self.assertEqual(len(self.f['data2'].dims[1]), 1)
        self.assertEqual(len(self.f['data2'].dims[2]), 2)

    def test_is_scale(self):
        """Test Dataset.is_scale property"""
        self.assertTrue(self.f['x1'].is_scale)
        self.assertTrue(self.f['x2'].is_scale)
        self.assertTrue(self.f['y1'].is_scale)
        self.assertFalse(self.f['z1'].is_scale)
        self.assertFalse(self.f['data'].is_scale)
        self.assertFalse(self.f['data2'].is_scale)