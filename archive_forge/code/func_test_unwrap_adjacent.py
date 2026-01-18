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
def test_unwrap_adjacent(self):
    self.assertEqual(_transform(FOOBAR, Transformer('foo|bar').unwrap()), [(None, START, u'root'), (None, TEXT, u'ROOT'), (INSIDE, TEXT, u'FOO'), (INSIDE, TEXT, u'BAR'), (None, END, u'root')])