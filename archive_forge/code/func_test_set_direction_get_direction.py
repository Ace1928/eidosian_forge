import os
import itertools
import logging
import pickle
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.core.base.suffix import (
from pyomo.environ import (
from io import StringIO
def test_set_direction_get_direction(self):
    model = ConcreteModel()
    model.junk = Suffix(direction=Suffix.LOCAL)
    self.assertEqual(model.junk.direction, Suffix.LOCAL)
    model.junk.direction = Suffix.EXPORT
    self.assertEqual(model.junk.direction, Suffix.EXPORT)
    model.junk.direction = Suffix.IMPORT
    self.assertEqual(model.junk.direction, Suffix.IMPORT)
    model.junk.direction = Suffix.IMPORT_EXPORT
    self.assertEqual(model.junk.direction, Suffix.IMPORT_EXPORT)
    with LoggingIntercept() as LOG:
        model.junk.set_direction(1)
    self.assertEqual(model.junk.direction, Suffix.EXPORT)
    self.assertRegex(LOG.getvalue().replace('\n', ' '), '^DEPRECATED: Suffix.set_direction is replaced with the Suffix.direction property')
    model.junk.direction = 'IMPORT'
    with LoggingIntercept() as LOG:
        self.assertEqual(model.junk.get_direction(), Suffix.IMPORT)
    self.assertRegex(LOG.getvalue().replace('\n', ' '), '^DEPRECATED: Suffix.get_direction is replaced with the Suffix.direction property')
    with self.assertRaisesRegex(ValueError, "'a' is not a valid SuffixDirection"):
        model.junk.direction = 'a'
    with self.assertRaisesRegex(ValueError, 'None is not a valid SuffixDirection'):
        model.junk.direction = None