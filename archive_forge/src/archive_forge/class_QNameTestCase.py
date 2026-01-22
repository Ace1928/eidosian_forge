import pickle
import unittest
from genshi import core
from genshi.core import Markup, Attrs, Namespace, QName, escape, unescape
from genshi.input import XML
from genshi.compat import StringIO, BytesIO, IS_PYTHON2
from genshi.tests.test_utils import doctest_suite
class QNameTestCase(unittest.TestCase):

    def test_pickle(self):
        qname = QName('http://www.example.org/namespace}elem')
        buf = BytesIO()
        pickle.dump(qname, buf, 2)
        buf.seek(0)
        unpickled = pickle.load(buf)
        self.assertEqual('{http://www.example.org/namespace}elem', unpickled)
        self.assertEqual('http://www.example.org/namespace', unpickled.namespace)
        self.assertEqual('elem', unpickled.localname)

    def test_repr(self):
        self.assertEqual("QName('elem')", repr(QName('elem')))
        self.assertEqual("QName('http://www.example.org/namespace}elem')", repr(QName('http://www.example.org/namespace}elem')))

    def test_repr_eval(self):
        qn = QName('elem')
        self.assertEqual(eval(repr(qn)), qn)

    def test_repr_eval_non_ascii(self):
        qn = QName(u'élem')
        self.assertEqual(eval(repr(qn)), qn)

    def test_leading_curly_brace(self):
        qname = QName('{http://www.example.org/namespace}elem')
        self.assertEqual('http://www.example.org/namespace', qname.namespace)
        self.assertEqual('elem', qname.localname)

    def test_curly_brace_equality(self):
        qname1 = QName('{http://www.example.org/namespace}elem')
        qname2 = QName('http://www.example.org/namespace}elem')
        self.assertEqual(qname1.namespace, qname2.namespace)
        self.assertEqual(qname1.localname, qname2.localname)
        self.assertEqual(qname1, qname2)