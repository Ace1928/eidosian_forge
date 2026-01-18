import warnings
import dis
from itertools import product
import numpy as np
from numba import njit, typed, objmode, prange
from numba.core.utils import PYVERSION
from numba.core import ir_utils, ir
from numba.core.errors import (
from numba.tests.support import (
def test_try_except_else(self):

    @njit
    def udt(x, y, z, p):
        print('A')
        if x:
            print('B')
            try:
                print('C')
                if y:
                    print('D')
                    raise MyError('y')
                print('E')
            except Exception:
                print('F')
                try:
                    print('H')
                    try:
                        print('I')
                        if z:
                            print('J')
                            raise MyError('z')
                        print('K')
                    except Exception:
                        print('L')
                    else:
                        print('M')
                except Exception:
                    print('N')
                else:
                    print('O')
                print('P')
            else:
                print('G')
            print('Q')
        print('R')
    cases = list(product([True, False], repeat=4))
    self.assertTrue(cases)
    for x, y, z, p in cases:
        self.check_compare(udt, udt.py_func, x=x, y=y, z=z, p=p)