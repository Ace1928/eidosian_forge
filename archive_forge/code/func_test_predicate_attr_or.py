import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_predicate_attr_or(self):
    xml = XML('<root><item/><item important="very"/></root>')
    self._test_eval('item[@urgent or @important]', input=xml, output='<item important="very"/>')
    self._test_eval('item[@urgent or @notso]', input=xml, output='')