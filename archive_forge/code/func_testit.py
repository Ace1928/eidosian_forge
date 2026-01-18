import sys
from sympy.core.cache import cacheit, cached_property, lazy_function
from sympy.testing.pytest import raises
@cacheit
def testit(x):
    return x