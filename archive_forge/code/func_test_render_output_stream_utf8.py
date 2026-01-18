import pickle
import unittest
from genshi import core
from genshi.core import Markup, Attrs, Namespace, QName, escape, unescape
from genshi.input import XML
from genshi.compat import StringIO, BytesIO, IS_PYTHON2
from genshi.tests.test_utils import doctest_suite
def test_render_output_stream_utf8(self):
    xml = XML('<li>Über uns</li>')
    strio = BytesIO()
    self.assertEqual(None, xml.render(encoding='utf-8', out=strio))
    self.assertEqual(u'<li>Über uns</li>'.encode('utf-8'), strio.getvalue())