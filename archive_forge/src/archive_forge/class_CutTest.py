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
class CutTest(unittest.TestCase, BufferTestMixin):
    operation = 'cut'

    def test_cut_element(self):
        self.assertEqual(self._apply('foo'), ([(None, START, u'root'), (None, TEXT, u'ROOT'), (None, START, u'bar'), (None, TEXT, u'BAR'), (None, END, u'bar'), (None, END, u'root')], [[(None, START, u'foo'), (None, TEXT, u'FOO'), (None, END, u'foo')]]))

    def test_cut_adjacent_elements(self):
        self.assertEqual(self._apply('foo|bar'), ([(None, START, u'root'), (None, TEXT, u'ROOT'), (BREAK, BREAK, None), (None, END, u'root')], [[(None, START, u'foo'), (None, TEXT, u'FOO'), (None, END, u'foo')], [(None, START, u'bar'), (None, TEXT, u'BAR'), (None, END, u'bar')]]))

    def test_cut_all(self):
        self.assertEqual(self._apply('*|text()'), ([(None, 'START', u'root'), ('BREAK', 'BREAK', None), ('BREAK', 'BREAK', None), (None, 'END', u'root')], [[(None, 'TEXT', u'ROOT')], [(None, 'START', u'foo'), (None, 'TEXT', u'FOO'), (None, 'END', u'foo')], [(None, 'START', u'bar'), (None, 'TEXT', u'BAR'), (None, 'END', u'bar')]]))

    def test_cut_text(self):
        self.assertEqual(self._apply('*/text()'), ([(None, 'START', u'root'), (None, 'TEXT', u'ROOT'), (None, 'START', u'foo'), (None, 'END', u'foo'), (None, 'START', u'bar'), (None, 'END', u'bar'), (None, 'END', u'root')], [[(None, 'TEXT', u'FOO')], [(None, 'TEXT', u'BAR')]]))

    def test_cut_context(self):
        self.assertEqual(self._apply('.')[1], [[(None, 'START', u'root'), (None, 'TEXT', u'ROOT'), (None, 'START', u'foo'), (None, 'TEXT', u'FOO'), (None, 'END', u'foo'), (None, 'START', u'bar'), (None, 'TEXT', u'BAR'), (None, 'END', u'bar'), (None, 'END', u'root')]])

    def test_cut_attribute(self):
        self.assertEqual(self._apply('foo/@name', with_attrs=True), ([(None, START, (u'root', {})), (None, TEXT, u'ROOT'), (None, START, (u'foo', {u'size': u'100'})), (None, TEXT, u'FOO'), (None, END, u'foo'), (None, START, (u'bar', {u'name': u'bar'})), (None, TEXT, u'BAR'), (None, END, u'bar'), (None, END, u'root')], [[(None, ATTR, {u'name': u'foo'})]]))

    def test_cut_attributes(self):
        self.assertEqual(self._apply('foo/@*', with_attrs=True), ([(None, START, (u'root', {})), (None, TEXT, u'ROOT'), (None, START, (u'foo', {})), (None, TEXT, u'FOO'), (None, END, u'foo'), (None, START, (u'bar', {u'name': u'bar'})), (None, TEXT, u'BAR'), (None, END, u'bar'), (None, END, u'root')], [[(None, ATTR, {u'name': u'foo', u'size': u'100'})]]))