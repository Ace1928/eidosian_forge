import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_predicate_attr_and_position(self):
    xml = XML('<root><foo/><foo id="a1"/><foo id="a2"/></root>')
    self._test_eval('*[@id][2]', input=xml, output='<foo id="a2"/>')