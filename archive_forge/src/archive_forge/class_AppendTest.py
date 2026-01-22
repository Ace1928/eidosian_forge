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
class AppendTest(unittest.TestCase, ContentTestMixin):
    operation = 'append'

    def test_append_element(self):
        self.assertEqual(self._apply('foo'), [(None, START, u'root'), (None, TEXT, u'ROOT'), (ENTER, START, u'foo'), (INSIDE, TEXT, u'FOO'), (None, TEXT, u'CONTENT 1'), (EXIT, END, u'foo'), (None, START, u'bar'), (None, TEXT, u'BAR'), (None, END, u'bar'), (None, END, u'root')])

    def test_append_text(self):
        self.assertEqual(self._apply('text()'), [(None, START, u'root'), (OUTSIDE, TEXT, u'ROOT'), (None, START, u'foo'), (None, TEXT, u'FOO'), (None, END, u'foo'), (None, START, u'bar'), (None, TEXT, u'BAR'), (None, END, u'bar'), (None, END, u'root')])

    def test_append_context(self):
        self.assertEqual(self._apply('.'), [(ENTER, START, u'root'), (INSIDE, TEXT, u'ROOT'), (INSIDE, START, u'foo'), (INSIDE, TEXT, u'FOO'), (INSIDE, END, u'foo'), (INSIDE, START, u'bar'), (INSIDE, TEXT, u'BAR'), (INSIDE, END, u'bar'), (None, TEXT, u'CONTENT 1'), (EXIT, END, u'root')])

    def test_append_text_context(self):
        self.assertEqual(self._apply('.', html='foo'), [(OUTSIDE, TEXT, u'foo')])

    def test_append_adjacent_elements(self):
        self.assertEqual(self._apply('*'), [(None, START, u'root'), (None, TEXT, u'ROOT'), (ENTER, START, u'foo'), (INSIDE, TEXT, u'FOO'), (None, TEXT, u'CONTENT 1'), (EXIT, END, u'foo'), (ENTER, START, u'bar'), (INSIDE, TEXT, u'BAR'), (None, TEXT, u'CONTENT 2'), (EXIT, END, u'bar'), (None, END, u'root')])

    def test_append_all(self):
        self.assertEqual(self._apply('*|text()'), [(None, START, u'root'), (OUTSIDE, TEXT, u'ROOT'), (ENTER, START, u'foo'), (INSIDE, TEXT, u'FOO'), (None, TEXT, u'CONTENT 1'), (EXIT, END, u'foo'), (ENTER, START, u'bar'), (INSIDE, TEXT, u'BAR'), (None, TEXT, u'CONTENT 2'), (EXIT, END, u'bar'), (None, END, u'root')])

    def test_append_with_callback(self):
        count = [0]

        def content():
            count[0] += 1
            yield ('%2i.' % count[0])
        self.assertEqual(self._apply('foo', content), [(None, 'START', u'root'), (None, 'TEXT', u'ROOT'), (ENTER, 'START', u'foo'), (INSIDE, 'TEXT', u'FOO'), (None, 'TEXT', u' 1.'), (EXIT, 'END', u'foo'), (None, 'START', u'bar'), (None, 'TEXT', u'BAR'), (None, 'END', u'bar'), (None, 'END', u'root')])