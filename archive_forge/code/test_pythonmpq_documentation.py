from fractions import Fraction
from decimal import Decimal
import pickle
from typing import Callable, List, Tuple, Type
from sympy.testing.pytest import raises
from sympy.external.pythonmpq import PythonMPQ

test_pythonmpq.py

Test the PythonMPQ class for consistency with gmpy2's mpq type. If gmpy2 is
installed run the same tests for both.
