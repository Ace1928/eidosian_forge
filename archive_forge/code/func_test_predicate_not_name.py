import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_predicate_not_name(self):
    xml = XML('<root><foo/><bar/></root>')
    self._test_eval('*[not(name()="foo")]', input=xml, output='<bar/>')