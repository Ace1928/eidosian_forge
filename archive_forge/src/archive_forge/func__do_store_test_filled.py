import unittest
from os.path import abspath, dirname, join
import errno
import os
def _do_store_test_filled(self, store):
    self.assertTrue(store.count() == 4)
    self.assertRaises(KeyError, lambda: store.get('plop2'))
    self.assertRaises(KeyError, lambda: store.delete('plop2'))
    self.assertTrue(store.exists('plop'))
    self.assertTrue(store.get('plop').get('name') == 'Hello')
    self.assertTrue(store.put('plop', name='World', age=1))
    self.assertTrue(store.get('plop').get('name') == 'World')
    self.assertTrue(store.exists('plop'))
    self.assertTrue(store.delete('plop'))
    self.assertRaises(KeyError, lambda: store.delete('plop'))
    self.assertRaises(KeyError, lambda: store.get('plop'))