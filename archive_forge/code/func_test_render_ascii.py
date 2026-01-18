import pickle
import unittest
from genshi import core
from genshi.core import Markup, Attrs, Namespace, QName, escape, unescape
from genshi.input import XML
from genshi.compat import StringIO, BytesIO, IS_PYTHON2
from genshi.tests.test_utils import doctest_suite
def test_render_ascii(self):
    xml = XML('<li>Über uns</li>')
    self.assertEqual(u'<li>&#220;ber uns</li>'.encode('ascii'), xml.render(encoding='ascii'))