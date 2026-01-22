import copy
import itertools
import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
import pyomo.core.base
from pyomo.core.base.util import flatten_tuple
from pyomo.environ import (
from pyomo.core.base.set import _AnySet, RangeDifferenceError
class TestSetErrors(PyomoModel):

    def test_membership(self):
        self.assertEqual(0 in Boolean, True)
        self.assertEqual(1 in Boolean, True)
        self.assertEqual(True in Boolean, True)
        self.assertEqual(False in Boolean, True)
        self.assertEqual(1.1 in Boolean, False)
        self.assertEqual(2 in Boolean, False)
        self.assertEqual(0 in Integers, True)
        self.assertEqual(1 in Integers, True)
        self.assertEqual(True in Integers, True)
        self.assertEqual(False in Integers, True)
        self.assertEqual(1.1 in Integers, False)
        self.assertEqual(2 in Integers, True)
        self.assertEqual(0 in Reals, True)
        self.assertEqual(1 in Reals, True)
        self.assertEqual(True in Reals, True)
        self.assertEqual(False in Reals, True)
        self.assertEqual(1.1 in Reals, True)
        self.assertEqual(2 in Reals, True)
        self.assertEqual(0 in Any, True)
        self.assertEqual(1 in Any, True)
        self.assertEqual(True in Any, True)
        self.assertEqual(False in Any, True)
        self.assertEqual(1.1 in Any, True)
        self.assertEqual(2 in Any, True)

    @unittest.skipIf(not _has_numpy, 'Numpy is not installed')
    def test_numpy_membership(self):
        self.assertEqual(numpy.int_(0) in Boolean, True)
        self.assertEqual(numpy.int_(1) in Boolean, True)
        self.assertEqual(numpy.bool_(True) in Boolean, True)
        self.assertEqual(numpy.bool_(False) in Boolean, True)
        self.assertEqual(numpy.float_(1.1) in Boolean, False)
        self.assertEqual(numpy.int_(2) in Boolean, False)
        self.assertEqual(numpy.int_(0) in Integers, True)
        self.assertEqual(numpy.int_(1) in Integers, True)
        self.assertEqual(numpy.bool_(True) in Integers, True)
        self.assertEqual(numpy.bool_(False) in Integers, True)
        self.assertEqual(numpy.float_(1.1) in Integers, False)
        self.assertEqual(numpy.int_(2) in Integers, True)
        self.assertEqual(numpy.int_(0) in Reals, True)
        self.assertEqual(numpy.int_(1) in Reals, True)
        self.assertEqual(numpy.bool_(True) in Reals, True)
        self.assertEqual(numpy.bool_(False) in Reals, True)
        self.assertEqual(numpy.float_(1.1) in Reals, True)
        self.assertEqual(numpy.int_(2) in Reals, True)
        self.assertEqual(numpy.int_(0) in Any, True)
        self.assertEqual(numpy.int_(1) in Any, True)
        self.assertEqual(numpy.bool_(True) in Any, True)
        self.assertEqual(numpy.bool_(False) in Any, True)
        self.assertEqual(numpy.float_(1.1) in Any, True)
        self.assertEqual(numpy.int_(2) in Any, True)

    def test_setargs1(self):
        try:
            a = Set()
            c = Set(a, foo=None)
            self.fail('test_setargs1 - expected error because of bad argument')
        except ValueError:
            pass

    def test_setargs2(self):
        a = Set()
        b = Set(a)
        with self.assertRaisesRegex(ValueError, 'Error retrieving component IndexedSet\\[None\\]: The component has not been constructed.'):
            c = Set(within=b, dimen=2)
            c.construct()
        a = Set()
        b = Set()
        c = Set(within=b, dimen=1)
        c.construct()
        self.assertEqual(c.domain, b)

    def test_setargs3(self):
        model = ConcreteModel()
        model.a = Set(dimen=1, initialize=(1, 2, 3))
        try:
            model.b = Set(dimen=2, initialize=(1, 2, 3))
            self.fail('test_setargs3 - expected error because dimen does not match set values')
        except ValueError:
            pass

    def test_setargs4(self):
        model = ConcreteModel()
        model.A = Set(initialize=[1])
        model.B = Set(model.A, initialize={1: [1]})
        try:
            model.C = Set(model.B)
            self.fail('test_setargs4 - expected error when passing in a set that is indexed')
        except TypeError:
            pass

    def test_setargs5(self):
        model = AbstractModel()
        model.A = Set()
        model.B = Set()
        model.C = model.A | model.B
        model.Z = Set(model.C)
        model.Y = RangeSet(model.C)
        model.X = Param(model.C, default=0.0)

    @unittest.skip('_verify was removed during the set rewrite')
    def test_verify(self):
        a = Set(initialize=[1, 2, 3])
        b = Set(within=a)
        try:
            b._verify(4)
            self.fail('test_verify - bad value was expected')
        except ValueError:
            pass
        c = Set()
        try:
            c._verify((1, 2))
            self.fail('test_verify - bad value was expected')
        except ValueError:
            pass
        c = Set(dimen=2)
        try:
            c._verify((1, 2, 3))
            self.fail('test_verify - bad value was expected')
        except ValueError:
            pass

    def test_construct(self):
        a = Set(initialize={1: 2, 3: 4})
        with self.assertRaisesRegex(KeyError, "Cannot treat the scalar component '[^']*' as an indexed component"):
            a.construct()
        a = Set(initialize={})
        a.construct()
        self.assertEqual(a, EmptySet)

        def init_fn(model):
            return []
        a = Set(initialize=init_fn)
        a.construct()
        self.assertEqual(a, EmptySet)

    def test_add(self):
        a = Set()
        a.construct()
        a.add(1)
        a.add('a')
        try:
            a.add({})
            self.fail('test_add - expected type error because {} is unhashable')
        except:
            pass

    def test_getitem(self):
        a = Set(initialize=[2, 3])
        with self.assertRaisesRegex(RuntimeError, '.*before it has been constructed'):
            a[0]
        a.construct()
        with self.assertRaisesRegex(IndexError, 'Accessing Pyomo Sets by position is 1-based'):
            a[0]
        self.assertEqual(a[1], 2)

    def test_eq(self):
        a = Set(dimen=1, name='a', initialize=[1, 2])
        a.construct()
        b = Set(dimen=2)
        b.construct()
        self.assertEqual(a == b, False)
        self.assertTrue(not a.__eq__(Boolean))
        self.assertTrue(not Boolean == a)

    def test_neq(self):
        a = Set(dimen=1, initialize=[1, 2])
        a.construct()
        b = Set(dimen=2)
        b.construct()
        self.assertEqual(a != b, True)
        self.assertTrue(a.__ne__(Boolean))
        self.assertTrue(Boolean != a)

    def test_contains(self):
        a = Set(initialize=[1, 3, 5, 7])
        a.construct()
        b = Set(initialize=[1, 3])
        b.construct()
        self.assertEqual(b in a, True)
        self.assertEqual(a in b, False)
        self.assertEqual(1 in Integers, True)
        self.assertEqual(1 in NonNegativeIntegers, True)

    def test_subset(self):
        self.assertTrue(Integers.issubset(Reals))

    def test_superset(self):
        self.assertTrue(Reals > Integers)
        self.assertTrue(Integers.issubset(Reals))
        a = Set(initialize=[1, 3, 5, 7])
        a.construct()
        b = Set(initialize=[1, 3])
        b.construct()
        self.assertEqual(a >= b, True)

    def test_lt(self):
        self.assertTrue(Integers < Reals)
        a = Set(initialize=[1, 3, 5, 7])
        a.construct()
        a < Reals
        b = Set(initialize=[1, 3, 5])
        b.construct()
        self.assertEqual(a < a, False)
        self.assertEqual(b < a, True)
        c = Set(initialize=[(1, 2)])
        c.construct()
        self.assertFalse(a < c)

    def test_gt(self):
        a = Set(initialize=[1, 3, 5, 7])
        a.construct()
        c = Set(initialize=[(1, 2)])
        c.construct()
        self.assertFalse(a > c)

    def test_or(self):
        a = Set(initialize=[1, 2, 3])
        c = Set(initialize=[(1, 2)])
        a.construct()
        c.construct()
        self.assertEqual(Reals | Integers, Reals)
        self.assertEqual(a | Integers, Integers)
        self.assertEqual(a | c, [1, 2, 3, (1, 2)])

    def test_and(self):
        a = Set(initialize=[1, 2, 3])
        c = Set(initialize=[(1, 2)])
        a.construct()
        c.construct()
        self.assertEqual(Reals & Integers, Integers)
        self.assertEqual(a & Integers, a)
        self.assertEqual(a & c, EmptySet)

    def test_xor(self):
        a = Set(initialize=[1, 2, 3])
        a.construct()
        c = Set(initialize=[(1, 2)])
        c.construct()
        X = Reals ^ Integers
        self.assertIn(0.5, X)
        self.assertNotIn(1, X)
        with self.assertRaisesRegex(RangeDifferenceError, 'We do not support subtracting an infinite discrete range \\[0:inf\\] from an infinite continuous range \\[-inf..inf\\]'):
            X < Reals
        self.assertEqual(a ^ Integers, Integers - a)
        self.assertEqual(a ^ c, SetOf([1, 2, 3, (1, 2)]))

    def test_sub(self):
        a = Set(initialize=[1, 2, 3])
        a.construct()
        c = Set(initialize=[(1, 2)])
        c.construct()
        X = Reals - Integers
        self.assertIn(0.5, X)
        self.assertNotIn(1, X)
        with self.assertRaisesRegex(RangeDifferenceError, 'We do not support subtracting an infinite discrete range \\[0:inf\\] from an infinite continuous range \\[-inf..inf\\]'):
            X < Reals
        self.assertEqual(a - Integers, EmptySet)
        self.assertEqual(a - c, a)

    def test_mul(self):
        a = Set(initialize=[1, 2, 3])
        c = Set(initialize=[(1, 2)])
        a.construct()
        c.construct()
        self.assertEqual((Reals * Integers).dimen, 2)
        self.assertEqual((a * Integers).dimen, 2)
        try:
            a * 1
            self.fail('test_mul - expected TypeError')
        except TypeError:
            pass
        b = a * c

    def test_arrayset_construct(self):

        def tmp_constructor(model, ctr, index):
            if ctr == 10:
                return Set.End
            else:
                return ctr
        a = Set(initialize=[1, 2, 3])
        a.construct()
        b = Set(a, initialize=tmp_constructor)
        try:
            b.construct({4: None})
            self.fail('test_arrayset_construct - expected KeyError')
        except KeyError:
            pass
        b._constructed = False
        b.construct()
        self.assertEqual(len(b), 3)
        for i in b:
            self.assertEqual(i in a, True)
        self.assertEqual(b[1], [1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(b[2], [1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(b[3], [1, 2, 3, 4, 5, 6, 7, 8, 9])
        b = Set(a, a, initialize=tmp_constructor)
        with self.assertRaisesRegex(TypeError, "'int' object is not iterable"):
            b.construct()

    def test_prodset(self):
        a = Set(initialize=[1, 2])
        a.construct()
        b = Set(initialize=[6, 7])
        b.construct()
        c = a * b
        c.construct()
        self.assertEqual((6, 2) in c, False)
        c = pyomo.core.base.set.SetProduct(a, b)
        c.virtual = True
        self.assertEqual((6, 2) in c, False)
        self.assertEqual((1, 7) in c, True)