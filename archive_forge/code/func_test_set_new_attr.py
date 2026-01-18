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
def test_set_new_attr(self):
    self.assertEqual(self._attr('foo', 'title', 'FOO'), [(None, START, (u'root', {})), (None, TEXT, u'ROOT'), (ENTER, START, (u'foo', {u'name': u'foo', u'title': 'FOO', u'size': '100'})), (INSIDE, TEXT, u'FOO'), (EXIT, END, u'foo'), (None, START, (u'bar', {u'name': u'bar'})), (None, TEXT, u'BAR'), (None, END, u'bar'), (None, END, u'root')])