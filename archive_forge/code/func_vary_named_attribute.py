from breezy.tests import TestCase, TestLoader, iter_suite_tests, multiply_tests
from breezy.tests.scenarios import (load_tests_apply_scenarios,
def vary_named_attribute(attr_name):
    """More sophisticated: vary a named parameter"""
    yield ('a', {attr_name: 'a'})
    yield ('b', {attr_name: 'b'})