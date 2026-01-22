import pickle
import unittest
from genshi import core
from genshi.core import Markup, Attrs, Namespace, QName, escape, unescape
from genshi.input import XML
from genshi.compat import StringIO, BytesIO, IS_PYTHON2
from genshi.tests.test_utils import doctest_suite
class NamespaceTestCase(unittest.TestCase):

    def test_repr(self):
        self.assertEqual("Namespace('http://www.example.org/namespace')", repr(Namespace('http://www.example.org/namespace')))

    def test_repr_eval(self):
        ns = Namespace('http://www.example.org/namespace')
        self.assertEqual(eval(repr(ns)), ns)

    def test_repr_eval_non_ascii(self):
        ns = Namespace(u'http://www.example.org/nämespäcé')
        self.assertEqual(eval(repr(ns)), ns)

    def test_pickle(self):
        ns = Namespace('http://www.example.org/namespace')
        buf = BytesIO()
        pickle.dump(ns, buf, 2)
        buf.seek(0)
        unpickled = pickle.load(buf)
        self.assertEqual("Namespace('http://www.example.org/namespace')", repr(unpickled))
        self.assertEqual('http://www.example.org/namespace', unpickled.uri)