from os import walk, sep, pardir
from os.path import split, join, abspath, exists, isfile
from glob import glob
import re
import random
import ast
from sympy.testing.pytest import raises
from sympy.testing.quality_unicode import _test_this_file_encoding
def test_test_unicode_encoding():
    unicode_whitelist = ['foo']
    unicode_strict_whitelist = ['bar']
    fname = 'abc'
    test_file = ['α']
    raises(AssertionError, lambda: _test_this_file_encoding(fname, test_file, unicode_whitelist, unicode_strict_whitelist))
    fname = 'abc'
    test_file = ['abc']
    _test_this_file_encoding(fname, test_file, unicode_whitelist, unicode_strict_whitelist)
    fname = 'foo'
    test_file = ['abc']
    raises(AssertionError, lambda: _test_this_file_encoding(fname, test_file, unicode_whitelist, unicode_strict_whitelist))
    fname = 'bar'
    test_file = ['abc']
    _test_this_file_encoding(fname, test_file, unicode_whitelist, unicode_strict_whitelist)