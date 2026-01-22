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
class SelectTest(unittest.TestCase):
    """Test .select()"""

    def _select(self, select):
        html = HTML(FOOBAR, encoding='utf-8')
        if isinstance(select, six.string_types):
            select = [select]
        transformer = Transformer(select[0])
        for sel in select[1:]:
            transformer = transformer.select(sel)
        return _transform(html, transformer)

    def test_select_single_element(self):
        self.assertEqual(self._select('foo'), [(None, START, u'root'), (None, TEXT, u'ROOT'), (ENTER, START, u'foo'), (INSIDE, TEXT, u'FOO'), (EXIT, END, u'foo'), (None, START, u'bar'), (None, TEXT, u'BAR'), (None, END, u'bar'), (None, END, u'root')])

    def test_select_context(self):
        self.assertEqual(self._select('.'), [(ENTER, START, u'root'), (INSIDE, TEXT, u'ROOT'), (INSIDE, START, u'foo'), (INSIDE, TEXT, u'FOO'), (INSIDE, END, u'foo'), (INSIDE, START, u'bar'), (INSIDE, TEXT, u'BAR'), (INSIDE, END, u'bar'), (EXIT, END, u'root')])

    def test_select_inside_select(self):
        self.assertEqual(self._select(['.', 'foo']), [(None, START, u'root'), (None, TEXT, u'ROOT'), (ENTER, START, u'foo'), (INSIDE, TEXT, u'FOO'), (EXIT, END, u'foo'), (None, START, u'bar'), (None, TEXT, u'BAR'), (None, END, u'bar'), (None, END, u'root')])

    def test_select_text(self):
        self.assertEqual(self._select('*/text()'), [(None, START, u'root'), (None, TEXT, u'ROOT'), (None, START, u'foo'), (OUTSIDE, TEXT, u'FOO'), (None, END, u'foo'), (None, START, u'bar'), (OUTSIDE, TEXT, u'BAR'), (None, END, u'bar'), (None, END, u'root')])

    def test_select_attr(self):
        self.assertEqual(self._select('foo/@name'), [(None, START, u'root'), (None, TEXT, u'ROOT'), (ATTR, ATTR, {'name': u'foo'}), (None, START, u'foo'), (None, TEXT, u'FOO'), (None, END, u'foo'), (None, START, u'bar'), (None, TEXT, u'BAR'), (None, END, u'bar'), (None, END, u'root')])

    def test_select_text_context(self):
        self.assertEqual(list(Transformer('.')(HTML(u'foo'), keep_marks=True)), [('OUTSIDE', ('TEXT', u'foo', (None, 1, 0)))])