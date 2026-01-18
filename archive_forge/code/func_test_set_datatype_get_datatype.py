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
def test_set_datatype_get_datatype(self):
    model = ConcreteModel()
    model.junk = Suffix(datatype=Suffix.FLOAT)
    self.assertEqual(model.junk.datatype, Suffix.FLOAT)
    model.junk.datatype = Suffix.INT
    self.assertEqual(model.junk.datatype, Suffix.INT)
    model.junk.datatype = None
    self.assertEqual(model.junk.datatype, None)
    model.junk.datatype = 'FLOAT'
    self.assertEqual(model.junk.datatype, Suffix.FLOAT)
    model.junk.datatype = 'INT'
    self.assertEqual(model.junk.datatype, Suffix.INT)
    model.junk.datatype = 4
    self.assertEqual(model.junk.datatype, Suffix.FLOAT)
    model.junk.datatype = 0
    self.assertEqual(model.junk.datatype, Suffix.INT)
    with LoggingIntercept() as LOG:
        model.junk.set_datatype(None)
    self.assertEqual(model.junk.datatype, None)
    self.assertRegex(LOG.getvalue().replace('\n', ' '), '^DEPRECATED: Suffix.set_datatype is replaced with the Suffix.datatype property')
    model.junk.datatype = 'FLOAT'
    with LoggingIntercept() as LOG:
        self.assertEqual(model.junk.get_datatype(), Suffix.FLOAT)
    self.assertRegex(LOG.getvalue().replace('\n', ' '), '^DEPRECATED: Suffix.get_datatype is replaced with the Suffix.datatype property')
    with self.assertRaisesRegex(ValueError, '1.0 is not a valid SuffixDataType'):
        model.junk.datatype = 1.0