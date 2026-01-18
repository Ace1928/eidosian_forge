import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_predicate_advanced_position(self):
    xml = XML('<root><a><b><c><d><e/></d></c></b></a></root>')
    self._test_eval('descendant-or-self::*/descendant-or-self::*/descendant-or-self::*[2]/self::*/descendant::*[3]', input=xml, output='<d><e/></d>')