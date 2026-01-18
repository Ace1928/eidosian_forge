import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_predicate_attr_greater_than(self):
    xml = XML('<root><item priority="3"/></root>')
    self._test_eval('item[@priority>3]', input=xml, output='')
    self._test_eval('item[@priority>2]', input=xml, output='<item priority="3"/>')