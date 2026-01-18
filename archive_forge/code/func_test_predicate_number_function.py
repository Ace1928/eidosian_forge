import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_predicate_number_function(self):
    xml = XML('<root><foo>bar</foo></root>')
    self._test_eval('*[number("3.0")=3]', input=xml, output='<foo>bar</foo>')
    self._test_eval('*[number("3.0")=3.0]', input=xml, output='<foo>bar</foo>')
    self._test_eval('*[number("0.1")=.1]', input=xml, output='<foo>bar</foo>')