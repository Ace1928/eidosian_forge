from os import walk, sep, pardir
from os.path import split, join, abspath, exists, isfile
from glob import glob
import re
import random
import ast
from sympy.testing.pytest import raises
from sympy.testing.quality_unicode import _test_this_file_encoding

    This test tests all files in SymPy and checks that:
      o no lines contains a trailing whitespace
      o no lines end with 

      o no line uses tabs instead of spaces
      o that the file ends with a single newline
      o there are no general or string exceptions
      o there are no old style raise statements
      o name of arg-less test suite functions start with _ or test_
      o no duplicate function names that start with test_
      o no assignments to self variable in class methods
      o no lines contain ".func is" except in the test suite
      o there is no do-nothing expression like `a == b` or `x + 1`
    