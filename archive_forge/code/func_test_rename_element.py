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
def test_rename_element(self):
    self.assertEqual(self._rename('foo|bar'), [(None, START, u'root'), (None, TEXT, u'ROOT'), (ENTER, START, u'foobar'), (INSIDE, TEXT, u'FOO'), (EXIT, END, u'foobar'), (ENTER, START, u'foobar'), (INSIDE, TEXT, u'BAR'), (EXIT, END, u'foobar'), (None, END, u'root')])