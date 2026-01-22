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
class ArraySet2(PyomoModel):

    def setUp(self):
        PyomoModel.setUp(self)
        OUTPUT = open(currdir + 'setA.dat', 'w')
        OUTPUT.write('data; set Z := A C; set Y := 1 2 ; set A[A,1] := 1 3 5 7; set A[C,2] := 3 5 7 9; end;')
        OUTPUT.close()
        self.model.Z = Set()
        self.model.Y = Set()
        self.model.A = Set(self.model.Z, self.model.Y)
        self.model.tmpset1 = Set()
        self.model.tmpset2 = Set()
        self.model.tmpset3 = Set()
        self.instance = self.model.create_instance(currdir + 'setA.dat')
        self.e1 = ('A1', 1)

    def test_bounds(self):
        self.assertEqual(self.instance.A['A', 1].bounds(), (1, 7))

    def test_getitem(self):
        """Check the access to items"""
        try:
            tmp = []
            for val in self.instance.A['A', 1]:
                tmp.append(val)
            tmp.sort()
        except:
            self.fail('Problems getting a valid subsetset from a set array')
        self.assertEqual(tmp, [1, 3, 5, 7])
        try:
            tmp = self.instance.A['A', 2]
        except:
            self.fail('Problems getting a valid uninitialized subset from a set array')
        try:
            tmp = self.instance.A['A', 3]
        except KeyError:
            pass
        else:
            self.fail('Problems getting an invalid set from a set array')

    def Xtest_setitem(self):
        """Check the access to items"""
        try:
            self.model.Y = Set(initialize=[1, 2])
            self.model.Z = Set(initialize=['A', 'C'])
            self.model.A = Set(self.model.Z, self.model.Y, initialize={'A': [1]})
            self.instance = self.model.create_instance()
            tmp = [1, 6, 9]
            self.instance.A['A'] = tmp
            self.instance.A['C'] = tmp
        except:
            self.fail('Problems setting a valid set into a set array')
        try:
            self.instance.A['D'] = tmp
        except KeyError:
            pass
        else:
            self.fail('Problems setting an invalid set into a set array')

    def Xtest_keys(self):
        """Check the keys for the array"""
        tmp = self.instance.A.keys()
        tmp.sort()
        self.assertEqual(tmp, ['A', 'C'])

    def Xtest_len(self):
        """Check that a simple set of numeric elements has the right size"""
        try:
            len(self.instance.A)
        except TypeError:
            pass
        else:
            self.fail('fail test_len')

    def Xtest_data(self):
        """Check that we can access the underlying set data"""
        try:
            self.instance.A.data()
        except TypeError:
            pass
        else:
            self.fail('fail test_data')

    def Xtest_dim(self):
        """Check that a simple set has dimension zero for its indexing"""
        self.assertEqual(self.instance.A.dim(), 1)

    def Xtest_clear(self):
        """Check the clear() method empties the set"""
        self.instance.A.clear()
        for key in self.instance.A:
            self.assertEqual(len(self.instance.A[key]), 0)

    def Xtest_virtual(self):
        """Check if this is not a virtual set"""
        self.assertEqual(self.instance.A.virtual, False)

    def Xtest_check_values(self):
        """Check if the values added to this set are valid"""
        self.instance.A.check_values()

    def Xtest_first(self):
        """Check that we can get the 'first' value in the set"""
        pass

    def Xtest_removeValid(self):
        """Check that we can remove a valid set element"""
        pass

    def Xtest_removeInvalid(self):
        """Check that we fail to remove an invalid set element"""
        pass

    def Xtest_discardValid(self):
        """Check that we can discard a valid set element"""
        pass

    def Xtest_discardInvalid(self):
        """Check that we fail to remove an invalid set element without an exception"""
        pass

    def Xtest_iterator(self):
        """Check that we can iterate through the set"""
        tmp = 0
        for key in self.instance.A:
            tmp += len(self.instance.A[key])
        self.assertEqual(tmp, 8)

    def Xtest_eq1(self):
        """Various checks for set equality and inequality (1)"""
        try:
            self.assertEqual(self.instance.A == self.instance.tmpset1, True)
            self.assertEqual(self.instance.tmpset1 == self.instance.A, True)
            self.assertEqual(self.instance.A != self.instance.tmpset1, False)
            self.assertEqual(self.instance.tmpset1 != self.instance.A, False)
        except TypeError:
            pass
        else:
            self.fail('fail test_eq1')

    def Xtest_eq2(self):
        """Various checks for set equality and inequality (2)"""
        try:
            self.assertEqual(self.instance.A == self.instance.tmpset2, False)
            self.assertEqual(self.instance.tmpset2 == self.instance.A, False)
            self.assertEqual(self.instance.A != self.instance.tmpset2, True)
            self.assertEqual(self.instance.tmpset2 != self.instance.A, True)
        except TypeError:
            pass
        else:
            self.fail('fail test_eq2')

    def Xtest_contains(self):
        """Various checks for contains() method"""
        tmp = self.e1 in self.instance.A
        self.assertEqual(tmp, False)

    def Xtest_or(self):
        """Check that set union works"""
        try:
            self.instance.A | self.instance.tmpset3
        except TypeError:
            pass
        else:
            self.fail('fail test_or')

    def Xtest_and(self):
        """Check that set intersection works"""
        try:
            self.instance.tmp = self.instance.A & self.instance.tmpset3
        except TypeError:
            pass
        else:
            self.fail('fail test_and')

    def Xtest_xor(self):
        """Check that set exclusive or works"""
        try:
            self.instance.A ^ self.instance.tmpset3
        except TypeError:
            pass
        else:
            self.fail('fail test_xor')

    def Xtest_diff(self):
        """Check that set difference works"""
        try:
            self.instance.A - self.instance.tmpset3
        except TypeError:
            pass
        else:
            self.fail('fail test_diff')

    def Xtest_mul(self):
        """Check that set cross-product works"""
        try:
            self.instance.A * self.instance.tmpset3
        except TypeError:
            pass
        else:
            self.fail('fail test_mul')