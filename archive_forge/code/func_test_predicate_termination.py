import unittest
from genshi.core import Attrs, QName
from genshi.input import XML
from genshi.path import Path, PathParser, PathSyntaxError, GenericStrategy, \
from genshi.tests.test_utils import doctest_suite
def test_predicate_termination(self):
    """
        Verify that a patch matching the self axis with a predicate doesn't
        cause an infinite loop. See <http://genshi.edgewall.org/ticket/82>.
        """
    xml = XML('<ul flag="1"><li>a</li><li>b</li></ul>')
    self._test_eval('.[@flag="1"]/*', input=xml, output='<li>a</li><li>b</li>')
    xml = XML('<ul flag="1"><li>a</li><li>b</li></ul>')
    self._test_eval('.[@flag="0"]/*', input=xml, output='')