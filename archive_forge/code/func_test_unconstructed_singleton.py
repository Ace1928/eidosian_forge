import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_unconstructed_singleton(self):
    a = Objective()
    self.assertEqual(a._constructed, False)
    self.assertEqual(len(a), 0)
    try:
        a()
        self.fail('Component is unconstructed')
    except ValueError:
        pass
    try:
        a.expr
        self.fail('Component is unconstructed')
    except ValueError:
        pass
    try:
        a.sense
        self.fail('Component is unconstructed')
    except ValueError:
        pass
    a.construct()
    a.set_sense(minimize)
    self.assertEqual(len(a), 1)
    self.assertEqual(a(), None)
    self.assertEqual(a.expr, None)
    self.assertEqual(a.sense, minimize)
    a.sense = maximize
    self.assertEqual(len(a), 1)
    self.assertEqual(a(), None)
    self.assertEqual(a.expr, None)
    self.assertEqual(a.sense, maximize)