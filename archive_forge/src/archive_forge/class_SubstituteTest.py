import doctest
import unittest
import six
from genshi import HTML
from genshi.builder import Element
from genshi.compat import IS_PYTHON2
from genshi.core import START, END, TEXT, QName, Attrs
from genshi.filters.transform import Transformer, StreamBuffer, ENTER, EXIT, \
import genshi.filters.transform
from genshi.tests.test_utils import doctest_suite
class SubstituteTest(unittest.TestCase):

    def _substitute(self, select, pattern, replace):
        return _transform(FOOBAR, Transformer(select).substitute(pattern, replace))

    def test_substitute_foo(self):
        self.assertEqual(self._substitute('foo', 'FOO|BAR', 'FOOOOO'), [(None, START, u'root'), (None, TEXT, u'ROOT'), (ENTER, START, u'foo'), (INSIDE, TEXT, u'FOOOOO'), (EXIT, END, u'foo'), (None, START, u'bar'), (None, TEXT, u'BAR'), (None, END, u'bar'), (None, END, u'root')])

    def test_substitute_foobar_with_group(self):
        self.assertEqual(self._substitute('foo|bar', '(FOO|BAR)', '(\\1)'), [(None, START, u'root'), (None, TEXT, u'ROOT'), (ENTER, START, u'foo'), (INSIDE, TEXT, u'(FOO)'), (EXIT, END, u'foo'), (ENTER, START, u'bar'), (INSIDE, TEXT, u'(BAR)'), (EXIT, END, u'bar'), (None, END, u'root')])