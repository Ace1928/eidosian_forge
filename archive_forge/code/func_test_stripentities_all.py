import pickle
import unittest
from genshi import core
from genshi.core import Markup, Attrs, Namespace, QName, escape, unescape
from genshi.input import XML
from genshi.compat import StringIO, BytesIO, IS_PYTHON2
from genshi.tests.test_utils import doctest_suite
def test_stripentities_all(self):
    markup = Markup('&amp; &#106;').stripentities()
    assert type(markup) is Markup
    self.assertEqual('& j', markup)