from collections import Counter
from pprint import pformat
from queue import Queue
import sys
import threading
import unittest
import testtools
def _flatten_tests(suite_or_case, unpack_outer=False):
    try:
        tests = iter(suite_or_case)
    except TypeError:
        return [(suite_or_case.id(), suite_or_case)]
    if type(suite_or_case) in (unittest.TestSuite,) or unpack_outer:
        result = []
        for test in tests:
            result.extend(_flatten_tests(test))
        return result
    else:
        suite_id = None
        tests = iterate_tests(suite_or_case)
        for test in tests:
            suite_id = test.id()
            break
        if hasattr(suite_or_case, 'sort_tests'):
            suite_or_case.sort_tests()
        return [(suite_id, suite_or_case)]