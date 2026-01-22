import h5py
from h5py._hl.files import _drivers
from h5py import File
from .common import ut, TestCase
import pytest
import io
import tempfile
import os
class TestDriverRegistration(TestCase):

    def test_register_driver(self):
        called_with = [None]

        def set_fapl(plist, *args, **kwargs):
            called_with[0] = (args, kwargs)
            return _drivers['sec2'](plist)
        h5py.register_driver('new-driver', set_fapl)
        self.assertIn('new-driver', h5py.registered_drivers())
        fname = self.mktemp()
        h5py.File(fname, driver='new-driver', driver_arg_0=0, driver_arg_1=1, mode='w')
        self.assertEqual(called_with, [((), {'driver_arg_0': 0, 'driver_arg_1': 1})])

    def test_unregister_driver(self):
        h5py.register_driver('new-driver', lambda plist: None)
        self.assertIn('new-driver', h5py.registered_drivers())
        h5py.unregister_driver('new-driver')
        self.assertNotIn('new-driver', h5py.registered_drivers())
        with self.assertRaises(ValueError) as e:
            fname = self.mktemp()
            h5py.File(fname, driver='new-driver', mode='w')
        self.assertEqual(str(e.exception), 'Unknown driver type "new-driver"')