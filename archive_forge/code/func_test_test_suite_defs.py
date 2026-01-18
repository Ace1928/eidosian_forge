from os import walk, sep, pardir
from os.path import split, join, abspath, exists, isfile
from glob import glob
import re
import random
import ast
from sympy.testing.pytest import raises
from sympy.testing.quality_unicode import _test_this_file_encoding
def test_test_suite_defs():
    candidates_ok = ['    def foo():\n', 'def foo(arg):\n', 'def _foo():\n', 'def test_foo():\n']
    candidates_fail = ['def foo():\n', 'def foo() :\n', 'def foo( ):\n', 'def  foo():\n']
    for c in candidates_ok:
        assert test_suite_def_re.search(c) is None, c
    for c in candidates_fail:
        assert test_suite_def_re.search(c) is not None, c