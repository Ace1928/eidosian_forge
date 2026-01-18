import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_sense_option(self):
    """Test sense option"""
    model = self.create_model()
    model.obj1 = Objective(model.A, model.A, rule=lambda m, i, j: 1.0, sense=maximize)
    model.obj2 = Objective(model.A, model.A, rule=lambda m, i, j: 1.0, sense=minimize)
    model.obj3 = Objective(model.A, model.A, rule=lambda m, i, j: 1.0)
    self.assertTrue(len(model.A) > 0)
    self.assertEqual(len(model.obj1), len(model.A) * len(model.A))
    self.assertEqual(len(model.obj2), len(model.A) * len(model.A))
    self.assertEqual(len(model.obj3), len(model.A) * len(model.A))
    for i in model.A:
        for j in model.A:
            self.assertEqual(model.obj1[i, j].sense, maximize)
            self.assertEqual(model.obj1[i, j].is_minimizing(), False)
            self.assertEqual(model.obj2[i, j].sense, minimize)
            self.assertEqual(model.obj2[i, j].is_minimizing(), True)
            self.assertEqual(model.obj3[i, j].sense, minimize)
            self.assertEqual(model.obj3[i, j].is_minimizing(), True)