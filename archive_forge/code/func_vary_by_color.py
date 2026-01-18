from breezy.tests import TestCase, TestLoader, iter_suite_tests, multiply_tests
from breezy.tests.scenarios import (load_tests_apply_scenarios,
def vary_by_color():
    """Very simple static variation example"""
    for color in ['red', 'green', 'blue']:
        yield (color, {'color': color})