import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_predicate_floor_function(self):
    xml = XML('<root><foo>bar</foo></root>')
    self._test_eval('*[floor("4.5")=4]', input=xml, output='<foo>bar</foo>')