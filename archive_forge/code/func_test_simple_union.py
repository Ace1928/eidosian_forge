import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_simple_union(self):
    xml = XML('<body>1<br />2<br />3<br /></body>')
    self._test_eval(path='*|text()', equiv='<Path "child::*|child::text()">', input=xml, output='1<br/>2<br/>3<br/>')