import pickle
import unittest
from genshi import core
from genshi.core import Markup, Attrs, Namespace, QName, escape, unescape
from genshi.input import XML
from genshi.compat import StringIO, BytesIO, IS_PYTHON2
from genshi.tests.test_utils import doctest_suite
class AttrsTestCase(unittest.TestCase):

    def test_pickle(self):
        attrs = Attrs([('attr1', 'foo'), ('attr2', 'bar')])
        buf = BytesIO()
        pickle.dump(attrs, buf, 2)
        buf.seek(0)
        unpickled = pickle.load(buf)
        self.assertEqual("Attrs([('attr1', 'foo'), ('attr2', 'bar')])", repr(unpickled))

    def test_non_ascii(self):
        attrs_tuple = Attrs([('attr1', u'föö'), ('attr2', u'bär')]).totuple()
        self.assertEqual(u'fööbär', attrs_tuple[1])