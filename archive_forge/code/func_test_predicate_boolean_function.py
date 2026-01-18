import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_predicate_boolean_function(self):
    xml = XML('<root><foo>bar</foo></root>')
    self._test_eval('*[boolean("")]', input=xml, output='')
    self._test_eval('*[boolean("yo")]', input=xml, output='<foo>bar</foo>')
    self._test_eval('*[boolean(0)]', input=xml, output='')
    self._test_eval('*[boolean(42)]', input=xml, output='<foo>bar</foo>')
    self._test_eval('*[boolean(false())]', input=xml, output='')
    self._test_eval('*[boolean(true())]', input=xml, output='<foo>bar</foo>')