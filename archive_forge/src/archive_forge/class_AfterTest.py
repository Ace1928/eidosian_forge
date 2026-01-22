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
class AfterTest(unittest.TestCase, ContentTestMixin):
    operation = 'after'

    def test_after_element(self):
        self.assertEqual(self._apply('foo'), [(None, START, u'root'), (None, TEXT, u'ROOT'), (ENTER, START, u'foo'), (INSIDE, TEXT, u'FOO'), (EXIT, END, u'foo'), (None, TEXT, u'CONTENT 1'), (None, START, u'bar'), (None, TEXT, u'BAR'), (None, END, u'bar'), (None, END, u'root')])

    def test_after_text(self):
        self.assertEqual(self._apply('text()'), [(None, START, u'root'), (OUTSIDE, TEXT, u'ROOT'), (None, TEXT, u'CONTENT 1'), (None, START, u'foo'), (None, TEXT, u'FOO'), (None, END, u'foo'), (None, START, u'bar'), (None, TEXT, u'BAR'), (None, END, u'bar'), (None, END, u'root')])

    def test_after_context(self):
        self.assertEqual(self._apply('.'), [(ENTER, START, u'root'), (INSIDE, TEXT, u'ROOT'), (INSIDE, START, u'foo'), (INSIDE, TEXT, u'FOO'), (INSIDE, END, u'foo'), (INSIDE, START, u'bar'), (INSIDE, TEXT, u'BAR'), (INSIDE, END, u'bar'), (EXIT, END, u'root'), (None, TEXT, u'CONTENT 1')])

    def test_after_text_context(self):
        self.assertEqual(self._apply('.', html='foo'), [(OUTSIDE, TEXT, u'foo'), (None, TEXT, u'CONTENT 1')])

    def test_after_adjacent_elements(self):
        self.assertEqual(self._apply('*'), [(None, START, u'root'), (None, TEXT, u'ROOT'), (ENTER, START, u'foo'), (INSIDE, TEXT, u'FOO'), (EXIT, END, u'foo'), (None, TEXT, u'CONTENT 1'), (ENTER, START, u'bar'), (INSIDE, TEXT, u'BAR'), (EXIT, END, u'bar'), (None, TEXT, u'CONTENT 2'), (None, END, u'root')])

    def test_after_all(self):
        self.assertEqual(self._apply('*|text()'), [(None, START, u'root'), (OUTSIDE, TEXT, u'ROOT'), (None, TEXT, u'CONTENT 1'), (ENTER, START, u'foo'), (INSIDE, TEXT, u'FOO'), (EXIT, END, u'foo'), (None, TEXT, u'CONTENT 2'), (ENTER, START, u'bar'), (INSIDE, TEXT, u'BAR'), (EXIT, END, u'bar'), (None, TEXT, u'CONTENT 3'), (None, END, u'root')])

    def test_after_with_callback(self):
        count = [0]

        def content():
            count[0] += 1
            yield ('%2i.' % count[0])
        self.assertEqual(self._apply('foo/text()', content), [(None, 'START', u'root'), (None, 'TEXT', u'ROOT'), (None, 'START', u'foo'), ('OUTSIDE', 'TEXT', u'FOO'), (None, 'TEXT', u' 1.'), (None, 'END', u'foo'), (None, 'START', u'bar'), (None, 'TEXT', u'BAR'), (None, 'END', u'bar'), (None, 'END', u'root')])