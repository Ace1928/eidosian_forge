from os import walk, sep, pardir
from os.path import split, join, abspath, exists, isfile
from glob import glob
import re
import random
import ast
from sympy.testing.pytest import raises
from sympy.testing.quality_unicode import _test_this_file_encoding
def test_test_duplicate_defs():
    candidates_ok = ['def foo():\ndef foo():\n', 'def test():\ndef test_():\n', 'def test_():\ndef test__():\n']
    candidates_fail = ['def test_():\ndef test_ ():\n', 'def test_1():\ndef  test_1():\n']
    ok = (None, 'check')

    def check(file):
        tests = 0
        test_set = set()
        for idx, line in enumerate(file.splitlines()):
            if test_ok_def_re.match(line):
                tests += 1
                test_set.add(line[3:].split('(')[0].strip())
                if len(test_set) != tests:
                    return (False, message_duplicate_test % ('check', idx + 1))
        return (None, 'check')
    for c in candidates_ok:
        assert check(c) == ok
    for c in candidates_fail:
        assert check(c) != ok