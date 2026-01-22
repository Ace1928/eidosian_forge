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
class InvertTest(unittest.TestCase):

    def _invert(self, select):
        return _transform(FOO, Transformer(select).invert())

    def test_invert_element(self):
        self.assertEqual(self._invert('foo'), [(OUTSIDE, START, u'root'), (OUTSIDE, TEXT, u'ROOT'), (None, START, u'foo'), (None, TEXT, u'FOO'), (None, END, u'foo'), (OUTSIDE, END, u'root')])

    def test_invert_inverted_element(self):
        self.assertEqual(_transform(FOO, Transformer('foo').invert().invert()), [(None, START, u'root'), (None, TEXT, u'ROOT'), (OUTSIDE, START, u'foo'), (OUTSIDE, TEXT, u'FOO'), (OUTSIDE, END, u'foo'), (None, END, u'root')])

    def test_invert_text(self):
        self.assertEqual(self._invert('foo/text()'), [(OUTSIDE, START, u'root'), (OUTSIDE, TEXT, u'ROOT'), (OUTSIDE, START, u'foo'), (None, TEXT, u'FOO'), (OUTSIDE, END, u'foo'), (OUTSIDE, END, u'root')])

    def test_invert_attribute(self):
        self.assertEqual(self._invert('foo/@name'), [(OUTSIDE, START, u'root'), (OUTSIDE, TEXT, u'ROOT'), (None, ATTR, {'name': u'foo'}), (OUTSIDE, START, u'foo'), (OUTSIDE, TEXT, u'FOO'), (OUTSIDE, END, u'foo'), (OUTSIDE, END, u'root')])

    def test_invert_context(self):
        self.assertEqual(self._invert('.'), [(None, START, u'root'), (None, TEXT, u'ROOT'), (None, START, u'foo'), (None, TEXT, u'FOO'), (None, END, u'foo'), (None, END, u'root')])

    def test_invert_text_context(self):
        self.assertEqual(_simplify(Transformer('.').invert()(HTML(u'foo'), keep_marks=True)), [(None, 'TEXT', u'foo')])