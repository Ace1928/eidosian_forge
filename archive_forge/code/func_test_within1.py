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
def test_within1(self):
    OUTPUT = open(currdir + 'setA.dat', 'w')
    OUTPUT.write('data; set Z := A C; set A[A] := 1 3 5 7.5; end;')
    OUTPUT.close()
    self.model.Z = Set()
    self.model.A = Set(self.model.Z, within=Integers)
    try:
        self.instance = self.model.create_instance(currdir + 'setA.dat')
    except ValueError:
        pass
    else:
        self.fail('fail test_within1')