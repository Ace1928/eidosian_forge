import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_predicate_attr_equality(self):
    xml = XML('<root><item/><item important="notso"/></root>')
    self._test_eval('item[@important="very"]', input=xml, output='')
    self._test_eval('item[@important!="very"]', input=xml, output='<item/><item important="notso"/>')