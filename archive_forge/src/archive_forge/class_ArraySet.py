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
class ArraySet(PyomoModel):

    def setUp(self):
        PyomoModel.setUp(self)
        OUTPUT = open(currdir + 'setA.dat', 'w')
        OUTPUT.write('data; set Z := A C; set A[A] := 1 3 5 7; set A[C] := 3 5 7 9; end;\n')
        OUTPUT.close()
        self.model.Z = Set()
        self.model.A = Set(self.model.Z)
        self.model.tmpset1 = Set()
        self.model.tmpset2 = Set()
        self.model.tmpset3 = Set()
        self.model.S = RangeSet(0, 5)
        self.model.T = RangeSet(0, 5)
        self.model.R = RangeSet(0, 3)
        self.model.Q_a = Set(initialize=[1, 3, 5, 7])
        self.model.Q_c = Set(initialize=[3, 5, 7, 9])
        self.instance = self.model.create_instance(currdir + 'setA.dat')
        self.e1 = ('A1', 1)

    def Xtest_bounds(self):
        self.assertEqual(self.instance.A.bounds(), None)

    def test_getitem(self):
        """Check the access to items"""
        try:
            tmp = []
            for val in self.instance.A['A']:
                tmp.append(val)
            tmp.sort()
        except:
            self.fail('Problems getting a valid set from a set array')
        self.assertEqual(tmp, [1, 3, 5, 7])
        try:
            tmp = self.instance.A['D']
        except KeyError:
            pass
        else:
            self.fail('Problems getting an invalid set from a set array')

    def test_setitem(self):
        """Check the access to items"""
        self.model.Z = Set(initialize=['A', 'C'])
        self.model.A = Set(self.model.Z, initialize={'A': [1]})
        self.instance = self.model.create_instance()
        tmp = [1, 6, 9]
        self.instance.A['A'] = tmp
        self.instance.A['C'] = tmp
        try:
            self.instance.A['D'] = tmp
        except KeyError:
            pass
        else:
            self.fail('Problems setting an invalid set into a set array')

    def test_keys(self):
        """Check the keys for the array"""
        tmp = list(self.instance.A.keys())
        tmp.sort()
        self.assertEqual(tmp, ['A', 'C'])

    def test_len(self):
        """Check that a simple set of numeric elements has the right size"""
        self.assertEqual(len(self.instance.A), 2)

    def test_data(self):
        """Check that we can access the underlying set data"""
        try:
            self.instance.A.data()
        except:
            self.fail('Expected data() method to pass')

    def test_dim(self):
        """Check that a simple set has dimension zero for its indexing"""
        self.assertEqual(self.instance.A.dim(), 1)

    def test_clear(self):
        """Check the clear() method empties the set"""
        self.instance.A.clear()
        for key in self.instance.A:
            self.assertEqual(len(self.instance.A[key]), 0)

    def test_virtual(self):
        """Check if this is not a virtual set"""
        with self.assertRaisesRegex(AttributeError, ".*no attribute 'virtual'"):
            self.instance.A.virtual

    def test_check_values(self):
        """Check if the values added to this set are valid"""
        self.assertTrue(self.instance.A.check_values())

    def test_first(self):
        """Check that we can get the 'first' value in the set"""
        pass

    def test_removeValid(self):
        """Check that we can remove a valid set element"""
        pass

    def test_removeInvalid(self):
        """Check that we fail to remove an invalid set element"""
        pass

    def test_discardValid(self):
        """Check that we can discard a valid set element"""
        pass

    def test_discardInvalid(self):
        """Check that we fail to remove an invalid set element without an exception"""
        pass

    def test_iterator(self):
        """Check that we can iterate through the set"""
        tmp = 0
        for key in self.instance.A:
            tmp += len(self.instance.A[key])
        self.assertEqual(tmp, 8)

    def test_eq1(self):
        """Various checks for set equality and inequality (1)"""
        self.assertEqual(self.instance.A != self.instance.tmpset1, True)
        self.assertEqual(self.instance.tmpset1 != self.instance.A, True)
        self.assertEqual(self.instance.A == self.instance.tmpset1, False)
        self.assertEqual(self.instance.tmpset1 == self.instance.A, False)

    def test_eq2(self):
        """Various checks for set equality and inequality (2)"""
        self.assertEqual(self.instance.A == self.instance.tmpset2, False)
        self.assertEqual(self.instance.tmpset2 == self.instance.A, False)
        self.assertEqual(self.instance.A != self.instance.tmpset2, True)
        self.assertEqual(self.instance.tmpset2 != self.instance.A, True)

    def test_eq3(self):
        """Various checks for set equality and inequality (3)"""
        self.assertEqual(self.instance.S == self.instance.S, True)
        self.assertEqual(self.instance.S != self.instance.S, False)
        self.assertEqual(self.instance.S == self.instance.T, True)
        self.assertEqual(self.instance.T == self.instance.S, True)
        self.assertEqual(self.instance.S != self.instance.R, True)
        self.assertEqual(self.instance.R != self.instance.S, True)
        self.assertEqual(self.instance.A['A'] == self.instance.Q_a, True)
        self.assertEqual(self.instance.Q_a == self.instance.A['A'], True)
        self.assertEqual(self.instance.A['C'] == self.instance.Q_c, True)
        self.assertEqual(self.instance.Q_c == self.instance.A['C'], True)
        self.assertEqual(self.instance.A == 1.0, False)
        self.assertEqual(1.0 == self.instance.A, False)
        self.assertEqual(self.instance.A != 1.0, True)
        self.assertEqual(1.0 != self.instance.A, True)

    def test_contains(self):
        """Various checks for contains() method"""
        tmp = self.e1 in self.instance.A
        self.assertEqual(tmp, False)

    def test_or(self):
        """Check that set union works"""
        with self.assertRaisesRegex(TypeError, 'Cannot apply a Set operator to an indexed Set component \\(A\\)'):
            self.instance.A | self.instance.tmpset3

    def test_and(self):
        """Check that set intersection works"""
        with self.assertRaisesRegex(TypeError, 'Cannot apply a Set operator to an indexed Set component \\(A\\)'):
            self.instance.A & self.instance.tmpset3

    def test_xor(self):
        """Check that set exclusive or works"""
        with self.assertRaisesRegex(TypeError, 'Cannot apply a Set operator to an indexed Set component \\(A\\)'):
            self.instance.A ^ self.instance.tmpset3

    def test_diff(self):
        """Check that set difference works"""
        with self.assertRaisesRegex(TypeError, 'Cannot apply a Set operator to an indexed Set component \\(A\\)'):
            self.instance.A - self.instance.tmpset3

    def test_mul(self):
        """Check that set cross-product works"""
        with self.assertRaisesRegex(TypeError, 'Cannot apply a Set operator to an indexed Set component \\(A\\)'):
            self.instance.A * self.instance.tmpset3

    def test_override_values(self):
        m = ConcreteModel()
        m.I = Set([1, 2, 3])
        m.I[1] = [1, 2, 3]
        self.assertEqual(sorted(m.I[1]), [1, 2, 3])
        m.I[1] = [4, 5, 6]
        self.assertEqual(sorted(m.I[1]), [4, 5, 6])
        m.J = Set([1, 2, 3], ordered=True)
        m.J[1] = [1, 3, 2]
        self.assertEqual(list(m.J[1]), [1, 3, 2])
        m.J[1] = [5, 4, 6]
        self.assertEqual(list(m.J[1]), [5, 4, 6])