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
class TestSetProduct(unittest.TestCase):

    def test_pickle(self):
        a = SetOf([1, 3, 5]) * SetOf([2, 3, 4])
        b = pickle.loads(pickle.dumps(a))
        self.assertIsNot(a, b)
        self.assertEqual(a, b)

    def test_bounds(self):
        a = SetOf([-2, -1, 0, 1])
        b = a * NonNegativeReals
        self.assertEqual(b.bounds(), ((-2, 0), (1, None)))
        c = a * RangeSet(3)
        self.assertEqual(c.bounds(), ((-2, 1), (1, 3)))

    def test_naming(self):
        m = ConcreteModel()
        m.I = SetOf([1, 2])
        a = m.I * [3, 4]
        b = [-1, 1] * a
        self.assertEqual(str(a), 'I*{3, 4}')
        self.assertEqual(str(b), '{-1, 1}*(I*{3, 4})')
        m.A = a
        self.assertEqual(str(a), 'A')
        self.assertEqual(str(b), '{-1, 1}*A')
        c = SetProduct(m.I, [1, 2], m.I)
        self.assertEqual(str(c), 'I*{1, 2}*I')

    def test_domain_and_pprint(self):
        m = ConcreteModel()
        m.I = SetOf([1, 2])
        m.A = m.I * [3, 4]
        self.assertIs(m.A._domain, m.A)
        m.A._domain = Any
        self.assertIs(m.A._domain, m.A)
        with self.assertRaisesRegex(ValueError, 'Setting the domain of a Set Operator is not allowed'):
            m.A._domain = None
        output = StringIO()
        m.A.pprint(ostream=output)
        ref = '\nA : Size=1, Index=None, Ordered=True\n    Key  : Dimen : Domain   : Size : Members\n    None :     2 : I*{3, 4} :    4 : {(1, 3), (1, 4), (2, 3), (2, 4)}\n'.strip()
        self.assertEqual(output.getvalue().strip(), ref)
        m = ConcreteModel()
        m.I = Set(initialize=[1, 2, 3])
        m.J = Reals * m.I
        output = StringIO()
        m.J.pprint(ostream=output)
        ref = '\nJ : Size=1, Index=None, Ordered=False\n    Key  : Dimen : Domain  : Size : Members\n    None :     2 : Reals*I :  Inf : <[-inf..inf], ([1], [2], [3])>\n'.strip()
        self.assertEqual(output.getvalue().strip(), ref)

    def test_dimen(self):
        m = ConcreteModel()
        m.I1 = SetOf([1, 2, 3, 4])
        m.I2 = SetOf([(1, 2), (3, 4)])
        m.IN = SetOf([(1, 2), (3, 4), 1, 2])
        m.J = Set()
        self.assertEqual((m.I1 * m.I1).dimen, 2)
        self.assertEqual((m.I2 * m.I2).dimen, 4)
        self.assertEqual((m.IN * m.IN).dimen, None)
        self.assertEqual((m.I1 * m.I2).dimen, 3)
        self.assertEqual((m.I2 * m.I1).dimen, 3)
        self.assertEqual((m.IN * m.I2).dimen, None)
        self.assertEqual((m.I2 * m.IN).dimen, None)
        self.assertEqual((m.IN * m.I1).dimen, None)
        self.assertEqual((m.I1 * m.IN).dimen, None)
        self.assertIs((m.J * m.I1).dimen, UnknownSetDimen)
        self.assertIs((m.J * m.I2).dimen, UnknownSetDimen)
        self.assertIs((m.J * m.IN).dimen, None)
        self.assertIs((m.I1 * m.J).dimen, UnknownSetDimen)
        self.assertIs((m.I2 * m.J).dimen, UnknownSetDimen)
        self.assertIs((m.IN * m.J).dimen, None)

    def test_cutPointGenerator(self):
        CG = SetProduct_InfiniteSet._cutPointGenerator
        i = Any
        j = SetOf([(1, 1), (1, 2), (2, 1), (2, 2)])
        test = list((tuple(_) for _ in CG((i, i), 3)))
        ref = [(0, 0, 3), (0, 1, 3), (0, 2, 3), (0, 3, 3)]
        self.assertEqual(test, ref)
        test = list((tuple(_) for _ in CG((i, i, i), 3)))
        ref = [(0, 0, 0, 3), (0, 0, 1, 3), (0, 0, 2, 3), (0, 0, 3, 3), (0, 1, 1, 3), (0, 1, 2, 3), (0, 1, 3, 3), (0, 2, 2, 3), (0, 2, 3, 3), (0, 3, 3, 3)]
        self.assertEqual(test, ref)
        test = list((tuple(_) for _ in CG((i, j, i), 5)))
        ref = [(0, 0, 2, 5), (0, 1, 3, 5), (0, 2, 4, 5), (0, 3, 5, 5)]
        self.assertEqual(test, ref)

    def test_subsets(self):
        a = SetOf([1])
        b = SetOf([1])
        c = SetOf([1])
        d = SetOf([1])
        x = a * b
        self.assertEqual(len(x._sets), 2)
        self.assertEqual(list(x.subsets()), [a, b])
        self.assertEqual(list(x.subsets(True)), [a, b])
        self.assertEqual(list(x.subsets(False)), [a, b])
        x = a * b * c
        self.assertEqual(len(x._sets), 2)
        self.assertEqual(list(x.subsets()), [a, b, c])
        self.assertEqual(list(x.subsets(True)), [a, b, c])
        self.assertEqual(list(x.subsets(False)), [a, b, c])
        x = a * b * (c * d)
        self.assertEqual(len(x._sets), 2)
        self.assertEqual(list(x.subsets()), [a, b, c, d])
        self.assertEqual(list(x.subsets(True)), [a, b, c, d])
        self.assertEqual(list(x.subsets(False)), [a, b, c, d])
        x = (a - b) * (c * d)
        self.assertEqual(len(x._sets), 2)
        self.assertEqual(len(list(x.subsets())), 3)
        self.assertEqual(len(list(x.subsets(False))), 3)
        self.assertEqual(list(x.subsets()), [a - b, c, d])
        self.assertEqual(len(list(x.subsets(True))), 4)
        self.assertEqual(list(x.subsets(True)), [a, b, c, d])

    def test_set_tuple(self):
        a = SetOf([1])
        b = SetOf([1])
        x = a * b
        os = StringIO()
        with LoggingIntercept(os, 'pyomo'):
            self.assertEqual(x.set_tuple, [a, b])
        self.assertRegex(os.getvalue(), '^DEPRECATED: SetProduct.set_tuple is deprecated.')

    def test_no_normalize_index(self):
        try:
            _oldFlatten = normalize_index.flatten
            I = SetOf([1, (1, 2)])
            J = SetOf([3, (2, 3)])
            x = I * J
            normalize_index.flatten = False
            self.assertIs(x.dimen, 2)
            self.assertIn(((1, 2), 3), x)
            self.assertIn((1, (2, 3)), x)
            self.assertNotIn((1, 2, 3), x)
            normalize_index.flatten = True
            self.assertIs(x.dimen, None)
            self.assertIn(((1, 2), 3), x)
            self.assertIn((1, (2, 3)), x)
            self.assertIn((1, 2, 3), x)
        finally:
            normalize_index.flatten = _oldFlatten

    def test_infinite_setproduct(self):
        x = PositiveIntegers * SetOf([2, 3, 5, 7])
        self.assertFalse(x.isfinite())
        self.assertFalse(x.isordered())
        self.assertIn((1, 2), x)
        self.assertNotIn((0, 2), x)
        self.assertNotIn((1, 1), x)
        self.assertNotIn(('a', 2), x)
        self.assertNotIn((2, 'a'), x)
        x = SetOf([2, 3, 5, 7]) * PositiveIntegers
        self.assertFalse(x.isfinite())
        self.assertFalse(x.isordered())
        self.assertIn((3, 2), x)
        self.assertNotIn((1, 2), x)
        self.assertNotIn((2, 0), x)
        self.assertNotIn(('a', 2), x)
        self.assertNotIn((2, 'a'), x)
        x = PositiveIntegers * PositiveIntegers
        self.assertFalse(x.isfinite())
        self.assertFalse(x.isordered())
        self.assertIn((3, 2), x)
        self.assertNotIn((0, 2), x)
        self.assertNotIn((2, 0), x)
        self.assertNotIn(('a', 2), x)
        self.assertNotIn((2, 'a'), x)

    def _verify_finite_product(self, a, b):
        if isinstance(a, (Set, SetOf, RangeSet)):
            a_ordered = a.isordered()
        else:
            a_ordered = type(a) is list
        if isinstance(b, (Set, SetOf, RangeSet)):
            b_ordered = b.isordered()
        else:
            b_ordered = type(b) is list
        self.assertFalse(a_ordered and b_ordered)
        x = a * b
        self.assertIs(type(x), SetProduct_FiniteSet)
        self.assertTrue(x.isfinite())
        self.assertFalse(x.isordered())
        self.assertEqual(len(x), 6)
        self.assertEqual(sorted(list(x)), [(1, 5), (1, 6), (2, 5), (2, 6), (3, 5), (3, 6)])
        self.assertEqual(x.ordered_data(), ((1, 5), (1, 6), (2, 5), (2, 6), (3, 5), (3, 6)))
        self.assertEqual(x.sorted_data(), ((1, 5), (1, 6), (2, 5), (2, 6), (3, 5), (3, 6)))
        self.assertNotIn(1, x)
        self.assertIn((1, 5), x)
        self.assertIn(((1,), 5), x)
        self.assertNotIn((1, 2, 3), x)
        self.assertNotIn((2, 4), x)

    def test_finite_setproduct(self):
        self._verify_finite_product(SetOf({3, 1, 2}), SetOf({6, 5}))
        self._verify_finite_product(SetOf({3, 1, 2}), SetOf([6, 5]))
        self._verify_finite_product(SetOf([3, 1, 2]), SetOf({6, 5}))
        self._verify_finite_product(SetOf([3, 1, 2]), {6, 5})
        self._verify_finite_product({3, 1, 2}, SetOf([6, 5]))
        self._verify_finite_product(SetOf({3, 1, 2}), [6, 5])
        self._verify_finite_product([3, 1, 2], SetOf({6, 5}))

    def _verify_ordered_product(self, a, b):
        if isinstance(a, (Set, SetOf, RangeSet)):
            a_ordered = a.isordered()
        else:
            a_ordered = type(a) is list
        self.assertTrue(a_ordered)
        if isinstance(b, (Set, SetOf, RangeSet)):
            b_ordered = b.isordered()
        else:
            b_ordered = type(b) is list
        self.assertTrue(b_ordered)
        x = a * b
        self.assertIs(type(x), SetProduct_OrderedSet)
        self.assertTrue(x.isfinite())
        self.assertTrue(x.isordered())
        self.assertEqual(len(x), 6)
        self.assertEqual(list(x), [(3, 6), (3, 5), (1, 6), (1, 5), (2, 6), (2, 5)])
        self.assertEqual(x.ordered_data(), ((3, 6), (3, 5), (1, 6), (1, 5), (2, 6), (2, 5)))
        self.assertEqual(x.sorted_data(), ((1, 5), (1, 6), (2, 5), (2, 6), (3, 5), (3, 6)))
        self.assertNotIn(1, x)
        self.assertIn((1, 5), x)
        self.assertIn(((1,), 5), x)
        self.assertNotIn((1, 2, 3), x)
        self.assertNotIn((2, 4), x)
        self.assertEqual(x.ord((3, 6)), 1)
        self.assertEqual(x.ord((3, 5)), 2)
        self.assertEqual(x.ord((1, 6)), 3)
        self.assertEqual(x.ord((1, 5)), 4)
        self.assertEqual(x.ord((2, 6)), 5)
        self.assertEqual(x.ord((2, 5)), 6)
        with self.assertRaisesRegex(IndexError, 'Cannot identify position of \\(3, 4\\) in Set SetProduct_OrderedSet'):
            x.ord((3, 4))
        self.assertEqual(x[1], (3, 6))
        self.assertEqual(x[2], (3, 5))
        self.assertEqual(x[3], (1, 6))
        self.assertEqual(x[4], (1, 5))
        self.assertEqual(x[5], (2, 6))
        self.assertEqual(x[6], (2, 5))
        with self.assertRaisesRegex(IndexError, 'SetProduct_OrderedSet index out of range'):
            x[7]
        self.assertEqual(x[-6], (3, 6))
        self.assertEqual(x[-5], (3, 5))
        self.assertEqual(x[-4], (1, 6))
        self.assertEqual(x[-3], (1, 5))
        self.assertEqual(x[-2], (2, 6))
        self.assertEqual(x[-1], (2, 5))
        with self.assertRaisesRegex(IndexError, 'SetProduct_OrderedSet index out of range'):
            x[-7]

    def test_ordered_setproduct(self):
        self._verify_ordered_product(SetOf([3, 1, 2]), SetOf([6, 5]))
        self._verify_ordered_product(SetOf([3, 1, 2]), [6, 5])
        self._verify_ordered_product([3, 1, 2], SetOf([6, 5]))

    def test_ordered_multidim_setproduct(self):
        x = SetOf([(1, 2), (3, 4)]) * SetOf([(5, 6), (7, 8)])
        self.assertEqual(x.dimen, 4)
        try:
            origFlattenCross = SetModule.FLATTEN_CROSS_PRODUCT
            SetModule.FLATTEN_CROSS_PRODUCT = True
            ref = [(1, 2, 5, 6), (1, 2, 7, 8), (3, 4, 5, 6), (3, 4, 7, 8)]
            self.assertEqual(list(x), ref)
            self.assertEqual(x.dimen, 4)
            SetModule.FLATTEN_CROSS_PRODUCT = False
            ref = [((1, 2), (5, 6)), ((1, 2), (7, 8)), ((3, 4), (5, 6)), ((3, 4), (7, 8))]
            self.assertEqual(list(x), ref)
            self.assertEqual(x.dimen, 2)
        finally:
            SetModule.FLATTEN_CROSS_PRODUCT = origFlattenCross
        self.assertIn(((1, 2), (5, 6)), x)
        self.assertIn((1, (2, 5), 6), x)
        self.assertIn((1, 2, 5, 6), x)
        self.assertNotIn((5, 6, 1, 2), x)

    def test_ordered_nondim_setproduct(self):
        NonDim = Set(initialize=[2, (2, 3)], dimen=None)
        NonDim.construct()
        NonDim2 = Set(initialize=[4, (3, 4)], dimen=None)
        NonDim2.construct()
        x = SetOf([1]).cross(NonDim, SetOf([3, 4, 5]))
        self.assertEqual(len(x), 6)
        try:
            origFlattenCross = SetModule.FLATTEN_CROSS_PRODUCT
            SetModule.FLATTEN_CROSS_PRODUCT = True
            ref = [(1, 2, 3), (1, 2, 4), (1, 2, 5), (1, 2, 3, 3), (1, 2, 3, 4), (1, 2, 3, 5)]
            self.assertEqual(list(x), ref)
            self.assertEqual(x.dimen, None)
            SetModule.FLATTEN_CROSS_PRODUCT = False
            ref = [(1, 2, 3), (1, 2, 4), (1, 2, 5), (1, (2, 3), 3), (1, (2, 3), 4), (1, (2, 3), 5)]
            self.assertEqual(list(x), ref)
            self.assertEqual(x.dimen, 3)
        finally:
            SetModule.FLATTEN_CROSS_PRODUCT = origFlattenCross
        self.assertIn((1, 2, 3), x)
        self.assertNotIn((1, 2, 6), x)
        self.assertIn((1, (2, 3), 3), x)
        self.assertIn((1, 2, 3, 3), x)
        self.assertNotIn((1, (2, 4), 3), x)
        self.assertEqual(x.ord((1, 2, 3)), 1)
        self.assertEqual(x.ord((1, (2, 3), 3)), 4)
        self.assertEqual(x.ord((1, (2, 3), 5)), 6)
        self.assertEqual(x.ord((1, 2, 3, 3)), 4)
        self.assertEqual(x.ord((1, 2, 3, 5)), 6)
        x = SetOf([1]).cross(NonDim, NonDim2, SetOf([0, 1]))
        self.assertEqual(len(x), 8)
        try:
            origFlattenCross = SetModule.FLATTEN_CROSS_PRODUCT
            SetModule.FLATTEN_CROSS_PRODUCT = True
            ref = [(1, 2, 4, 0), (1, 2, 4, 1), (1, 2, 3, 4, 0), (1, 2, 3, 4, 1), (1, 2, 3, 4, 0), (1, 2, 3, 4, 1), (1, 2, 3, 3, 4, 0), (1, 2, 3, 3, 4, 1)]
            self.assertEqual(list(x), ref)
            for i, v in enumerate(ref):
                self.assertEqual(x[i + 1], v)
            self.assertEqual(x.dimen, None)
            SetModule.FLATTEN_CROSS_PRODUCT = False
            ref = [(1, 2, 4, 0), (1, 2, 4, 1), (1, 2, (3, 4), 0), (1, 2, (3, 4), 1), (1, (2, 3), 4, 0), (1, (2, 3), 4, 1), (1, (2, 3), (3, 4), 0), (1, (2, 3), (3, 4), 1)]
            self.assertEqual(list(x), ref)
            for i, v in enumerate(ref):
                self.assertEqual(x[i + 1], v)
            self.assertEqual(x.dimen, 4)
        finally:
            SetModule.FLATTEN_CROSS_PRODUCT = origFlattenCross
        self.assertIn((1, 2, 4, 0), x)
        self.assertNotIn((1, 2, 6), x)
        self.assertIn((1, (2, 3), 4, 0), x)
        self.assertIn((1, 2, (3, 4), 0), x)
        self.assertIn((1, 2, 3, 4, 0), x)
        self.assertNotIn((1, 2, 5, 4, 0), x)
        self.assertEqual(x.ord((1, 2, 4, 0)), 1)
        self.assertEqual(x.ord((1, (2, 3), 4, 0)), 5)
        self.assertEqual(x.ord((1, 2, (3, 4), 0)), 3)
        self.assertEqual(x.ord((1, 2, 3, 4, 0)), 3)

    def test_setproduct_construct_data(self):
        m = AbstractModel()
        m.I = Set(initialize=[1, 2])
        m.J = m.I * m.I
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            m.create_instance(data={None: {'J': {None: [(1, 1), (1, 2), (2, 1), (2, 2)]}}})
        self.assertRegex(output.getvalue().replace('\n', ' '), '^DEPRECATED: Providing construction data to SetOperator objects is deprecated')
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            with self.assertRaisesRegex(ValueError, 'Constructing SetOperator J with incompatible data \\(data=\\{None: \\[\\(1, 1\\), \\(1, 2\\), \\(2, 1\\)\\]\\}'):
                m.create_instance(data={None: {'J': {None: [(1, 1), (1, 2), (2, 1)]}}})
        self.assertRegex(output.getvalue().replace('\n', ' '), '^DEPRECATED: Providing construction data to SetOperator objects is deprecated')
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core'):
            with self.assertRaisesRegex(ValueError, 'Constructing SetOperator J with incompatible data \\(data=\\{None: \\[\\(1, 3\\), \\(1, 2\\), \\(2, 1\\), \\(2, 2\\)\\]\\}'):
                m.create_instance(data={None: {'J': {None: [(1, 3), (1, 2), (2, 1), (2, 2)]}}})
        self.assertRegex(output.getvalue().replace('\n', ' '), '^DEPRECATED: Providing construction data to SetOperator objects is deprecated')

    def test_setproduct_nondim_set(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1, 2, 3])
        m.J = Set()
        m.K = Set(initialize=[4, 5, 6])
        m.Z = m.I * m.J * m.K
        self.assertEqual(len(m.Z), 0)
        self.assertNotIn((2, 5), m.Z)
        m.J.add(0)
        self.assertEqual(len(m.Z), 9)
        self.assertIn((2, 0, 5), m.Z)

    def test_setproduct_toolong_val(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1, 2, 3])
        m.J = Set(initialize=[4, 5, 6])
        m.Z = m.I * m.J
        self.assertIn((2, 5), m.Z)
        self.assertNotIn((2, 5, 3), m.Z)
        m = ConcreteModel()
        m.I = Set(initialize=[1, 2, 3])
        m.J = Set(initialize=[4, 5, 6], dimen=None)
        m.Z = m.I * m.J
        self.assertIn((2, 5), m.Z)
        self.assertNotIn((2, 5, 3), m.Z)