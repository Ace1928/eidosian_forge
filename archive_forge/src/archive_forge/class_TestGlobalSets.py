import copy
import itertools
import logging
import pickle
from io import StringIO
from collections import namedtuple as NamedTuple
import pyomo.common.unittest as unittest
from pyomo.common import DeveloperError
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import pandas as pd, pandas_available
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr import native_numeric_types, native_types
import pyomo.core.base.set as SetModule
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.initializer import (
from pyomo.core.base.set import (
from pyomo.environ import (
class TestGlobalSets(unittest.TestCase):

    def test_globals(self):
        self.assertEqual(Reals.__class__.__name__, 'GlobalSet')
        self.assertIsInstance(Reals, RangeSet)

    def test_pickle(self):
        a = pickle.loads(pickle.dumps(Reals))
        self.assertIs(a, Reals)

    def test_deepcopy(self):
        a = copy.deepcopy(Reals)
        self.assertIs(a, Reals)

    def test_name(self):
        self.assertEqual(str(Reals), 'Reals')
        self.assertEqual(str(Integers), 'Integers')

    def test_block_independent(self):
        m = ConcreteModel()
        with self.assertRaisesRegex(RuntimeError, "Cannot assign a GlobalSet 'Reals' to model 'unknown'"):
            m.a_set = Reals
        self.assertEqual(str(Reals), 'Reals')
        self.assertIsNone(Reals._parent)
        m.blk = Block()
        with self.assertRaisesRegex(RuntimeError, "Cannot assign a GlobalSet 'Reals' to block 'blk'"):
            m.blk.a_set = Reals
        self.assertEqual(str(Reals), 'Reals')
        self.assertIsNone(Reals._parent)

    def test_iteration(self):
        with self.assertRaisesRegex(TypeError, "'GlobalSet' object is not iterable \\(non-finite Set 'Reals' is not iterable\\)"):
            iter(Reals)
        with self.assertRaisesRegex(TypeError, "'GlobalSet' object is not iterable \\(non-finite Set 'Integers' is not iterable\\)"):
            iter(Integers)
        self.assertEqual(list(iter(Binary)), [0, 1])

    def test_declare(self):
        NS = {}
        DeclareGlobalSet(RangeSet(name='TrinarySet', ranges=(NR(0, 2, 1),)), NS)
        self.assertEqual(list(NS['TrinarySet']), [0, 1, 2])
        a = pickle.loads(pickle.dumps(NS['TrinarySet']))
        self.assertIs(a, NS['TrinarySet'])
        with self.assertRaisesRegex(NameError, "name 'TrinarySet' is not defined"):
            TrinarySet
        del SetModule.GlobalSets['TrinarySet']
        del NS['TrinarySet']
        DeclareGlobalSet(RangeSet(name='TrinarySet', ranges=(NR(0, 2, 1),)))
        self.assertEqual(list(TrinarySet), [0, 1, 2])
        a = pickle.loads(pickle.dumps(TrinarySet))
        self.assertIs(a, TrinarySet)
        del SetModule.GlobalSets['TrinarySet']
        del globals()['TrinarySet']
        with self.assertRaisesRegex(NameError, "name 'TrinarySet' is not defined"):
            TrinarySet

    def test_exceptions(self):
        with self.assertRaisesRegex(RuntimeError, 'Duplicate Global Set declaration, Reals'):
            DeclareGlobalSet(RangeSet(name='Reals', ranges=(NR(0, 2, 1),)))
        a = Reals
        DeclareGlobalSet(Reals)
        self.assertIs(a, Reals)
        self.assertIs(a, globals()['Reals'])
        self.assertIs(a, SetModule.GlobalSets['Reals'])
        NS = {}
        ts = DeclareGlobalSet(RangeSet(name='TrinarySet', ranges=(NR(0, 2, 1),)), NS)
        self.assertIs(NS['TrinarySet'], ts)
        DeclareGlobalSet(ts, NS)
        self.assertIs(NS['TrinarySet'], ts)
        NS['foo'] = None
        with self.assertRaisesRegex(RuntimeError, 'Refusing to overwrite global object, foo'):
            DeclareGlobalSet(RangeSet(name='foo', ranges=(NR(0, 2, 1),)), NS)

    def test_RealSet_IntegerSet(self):
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            a = SetModule.RealSet()
        self.assertIn('DEPRECATED: The use of RealSet,', output.getvalue())
        self.assertEqual(a, Reals)
        self.assertIsNot(a, Reals)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            a = SetModule.RealSet(bounds=(1, 3))
        self.assertIn('DEPRECATED: The use of RealSet,', output.getvalue())
        self.assertEqual(a.bounds(), (1, 3))
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            a = SetModule.IntegerSet()
        self.assertIn('DEPRECATED: The use of RealSet,', output.getvalue())
        self.assertEqual(a, Integers)
        self.assertIsNot(a, Integers)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            a = SetModule.IntegerSet(bounds=(1, 3))
        self.assertIn('DEPRECATED: The use of RealSet,', output.getvalue())
        self.assertEqual(a.bounds(), (1, 3))
        self.assertEqual(list(a), [1, 2, 3])
        m = ConcreteModel()
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            m.x = Var(within=SetModule.RealSet)
        self.assertIn('DEPRECATED: The use of RealSet,', output.getvalue())
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            m.y = Var(within=SetModule.RealSet())
        self.assertIn('DEPRECATED: The use of RealSet,', output.getvalue())
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            m.z = Var(within=SetModule.RealSet(bounds=(0, None)))
        self.assertIn('DEPRECATED: The use of RealSet,', output.getvalue())
        with self.assertRaisesRegex(RuntimeError, "Unexpected keyword arguments: \\{'foo': 5\\}"):
            IntegerSet(foo=5)

    def test_intervals(self):
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            a = SetModule.RealInterval()
        self.assertIn('RealInterval has been deprecated.', output.getvalue())
        self.assertEqual(a, Reals)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            a = SetModule.RealInterval(bounds=(0, None))
        self.assertIn('RealInterval has been deprecated.', output.getvalue())
        self.assertEqual(a, NonNegativeReals)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            a = SetModule.RealInterval(bounds=5)
        self.assertIn('RealInterval has been deprecated.', output.getvalue())
        self.assertEqual(a, RangeSet(1, 5, 0))
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            a = SetModule.RealInterval(bounds=(5,))
        self.assertIn('RealInterval has been deprecated.', output.getvalue())
        self.assertEqual(a, RangeSet(1, 5, 0))
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            a = SetModule.IntegerInterval()
        self.assertIn('IntegerInterval has been deprecated.', output.getvalue())
        self.assertEqual(a, Integers)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            a = SetModule.IntegerInterval(bounds=(0, None))
        self.assertIn('IntegerInterval has been deprecated.', output.getvalue())
        self.assertEqual(a, NonNegativeIntegers)
        self.assertFalse(a.isfinite())
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            a = SetModule.IntegerInterval(bounds=(None, -1))
        self.assertIn('IntegerInterval has been deprecated.', output.getvalue())
        self.assertEqual(a, NegativeIntegers)
        self.assertFalse(a.isfinite())
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            a = SetModule.IntegerInterval(bounds=(-float('inf'), -1))
        self.assertIn('IntegerInterval has been deprecated.', output.getvalue())
        self.assertEqual(a, NegativeIntegers)
        self.assertFalse(a.isfinite())
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            a = SetModule.IntegerInterval(bounds=(0, 3))
        self.assertIn('IntegerInterval has been deprecated.', output.getvalue())
        self.assertEqual(list(a), [0, 1, 2, 3])
        self.assertTrue(a.isfinite())
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            a = SetModule.IntegerInterval(bounds=5)
        self.assertIn('IntegerInterval has been deprecated.', output.getvalue())
        self.assertEqual(list(a), [1, 2, 3, 4, 5])
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            a = SetModule.IntegerInterval(bounds=(5,))
        self.assertIn('IntegerInterval has been deprecated.', output.getvalue())
        self.assertEqual(list(a), [1, 2, 3, 4, 5])