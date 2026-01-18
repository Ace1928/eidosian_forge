import sympy
import tempfile
import os
from sympy.core.mod import Mod
from sympy.core.relational import Eq
from sympy.core.symbol import symbols
from sympy.external import import_module
from sympy.tensor import IndexedBase, Idx
from sympy.utilities.autowrap import autowrap, ufuncify, CodeWrapError
from sympy.testing.pytest import skip
def test_wrap_twice_c_cython():
    has_module('Cython')
    runtest_autowrap_twice('C', 'cython')