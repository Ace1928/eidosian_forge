import pickle
import os
from os.path import abspath, dirname, join
import platform
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.environ import (
def verifyModel(self, ref, new):
    self.assertEqual(sorted(ref._data.keys()), sorted(new._data.keys()))
    for idx in ref._data.keys():
        self.assertEqual(type(ref._data[idx]), type(new._data[idx]))
        if idx is not None:
            self.assertNotEqual(id(ref._data[idx]), id(new._data[idx]))
    self.assertEqual(id(ref.solutions._instance()), id(ref))
    self.assertEqual(id(new.solutions._instance()), id(new))
    for idx in ref._data.keys():
        ref_c = ref._data[idx].component_map()
        new_c = new._data[idx].component_map()
        self.assertEqual(sorted(ref_c.keys()), sorted(new_c.keys()))
        for a in ref_c.keys():
            self.assertEqual(type(ref_c[a]), type(new_c[a]))
            self.assertNotEqual(id(ref_c[a]), id(new_c[a]))