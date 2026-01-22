from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
class SerializerTests(unittest.TestCase):

    def testNoNamespace(self):
        e = domish.Element((None, 'foo'))
        self.assertEqual(e.toXml(), '<foo/>')
        self.assertEqual(e.toXml(closeElement=0), '<foo>')

    def testDefaultNamespace(self):
        e = domish.Element(('testns', 'foo'))
        self.assertEqual(e.toXml(), "<foo xmlns='testns'/>")

    def testOtherNamespace(self):
        e = domish.Element(('testns', 'foo'), 'testns2')
        self.assertEqual(e.toXml({'testns': 'bar'}), "<bar:foo xmlns:bar='testns' xmlns='testns2'/>")

    def testChildDefaultNamespace(self):
        e = domish.Element(('testns', 'foo'))
        e.addElement('bar')
        self.assertEqual(e.toXml(), "<foo xmlns='testns'><bar/></foo>")

    def testChildSameNamespace(self):
        e = domish.Element(('testns', 'foo'))
        e.addElement(('testns', 'bar'))
        self.assertEqual(e.toXml(), "<foo xmlns='testns'><bar/></foo>")

    def testChildSameDefaultNamespace(self):
        e = domish.Element(('testns', 'foo'))
        e.addElement('bar', 'testns')
        self.assertEqual(e.toXml(), "<foo xmlns='testns'><bar/></foo>")

    def testChildOtherDefaultNamespace(self):
        e = domish.Element(('testns', 'foo'))
        e.addElement(('testns2', 'bar'), 'testns2')
        self.assertEqual(e.toXml(), "<foo xmlns='testns'><bar xmlns='testns2'/></foo>")

    def testOnlyChildDefaultNamespace(self):
        e = domish.Element((None, 'foo'))
        e.addElement(('ns2', 'bar'), 'ns2')
        self.assertEqual(e.toXml(), "<foo><bar xmlns='ns2'/></foo>")

    def testOnlyChildDefaultNamespace2(self):
        e = domish.Element((None, 'foo'))
        e.addElement('bar')
        self.assertEqual(e.toXml(), '<foo><bar/></foo>')

    def testChildInDefaultNamespace(self):
        e = domish.Element(('testns', 'foo'), 'testns2')
        e.addElement(('testns2', 'bar'))
        self.assertEqual(e.toXml(), "<xn0:foo xmlns:xn0='testns' xmlns='testns2'><bar/></xn0:foo>")

    def testQualifiedAttribute(self):
        e = domish.Element((None, 'foo'), attribs={('testns2', 'bar'): 'baz'})
        self.assertEqual(e.toXml(), "<foo xmlns:xn0='testns2' xn0:bar='baz'/>")

    def testQualifiedAttributeDefaultNS(self):
        e = domish.Element(('testns', 'foo'), attribs={('testns', 'bar'): 'baz'})
        self.assertEqual(e.toXml(), "<foo xmlns='testns' xmlns:xn0='testns' xn0:bar='baz'/>")

    def testTwoChilds(self):
        e = domish.Element(('', 'foo'))
        child1 = e.addElement(('testns', 'bar'), 'testns2')
        child1.addElement(('testns2', 'quux'))
        child2 = e.addElement(('testns3', 'baz'), 'testns4')
        child2.addElement(('testns', 'quux'))
        self.assertEqual(e.toXml(), "<foo><xn0:bar xmlns:xn0='testns' xmlns='testns2'><quux/></xn0:bar><xn1:baz xmlns:xn1='testns3' xmlns='testns4'><xn0:quux xmlns:xn0='testns'/></xn1:baz></foo>")

    def testXMLNamespace(self):
        e = domish.Element((None, 'foo'), attribs={('http://www.w3.org/XML/1998/namespace', 'lang'): 'en_US'})
        self.assertEqual(e.toXml(), "<foo xml:lang='en_US'/>")

    def testQualifiedAttributeGivenListOfPrefixes(self):
        e = domish.Element((None, 'foo'), attribs={('testns2', 'bar'): 'baz'})
        self.assertEqual(e.toXml({'testns2': 'qux'}), "<foo xmlns:qux='testns2' qux:bar='baz'/>")

    def testNSPrefix(self):
        e = domish.Element((None, 'foo'), attribs={('testns2', 'bar'): 'baz'})
        c = e.addElement(('testns2', 'qux'))
        c['testns2', 'bar'] = 'quux'
        self.assertEqual(e.toXml(), "<foo xmlns:xn0='testns2' xn0:bar='baz'><xn0:qux xn0:bar='quux'/></foo>")

    def testDefaultNSPrefix(self):
        e = domish.Element((None, 'foo'), attribs={('testns2', 'bar'): 'baz'})
        c = e.addElement(('testns2', 'qux'))
        c['testns2', 'bar'] = 'quux'
        c.addElement('foo')
        self.assertEqual(e.toXml(), "<foo xmlns:xn0='testns2' xn0:bar='baz'><xn0:qux xn0:bar='quux'><xn0:foo/></xn0:qux></foo>")

    def testPrefixScope(self):
        e = domish.Element(('testns', 'foo'))
        self.assertEqual(e.toXml(prefixes={'testns': 'bar'}, prefixesInScope=['bar']), '<bar:foo/>')

    def testLocalPrefixes(self):
        e = domish.Element(('testns', 'foo'), localPrefixes={'bar': 'testns'})
        self.assertEqual(e.toXml(), "<bar:foo xmlns:bar='testns'/>")

    def testLocalPrefixesWithChild(self):
        e = domish.Element(('testns', 'foo'), localPrefixes={'bar': 'testns'})
        e.addElement('baz')
        self.assertIdentical(e.baz.defaultUri, None)
        self.assertEqual(e.toXml(), "<bar:foo xmlns:bar='testns'><baz/></bar:foo>")

    def test_prefixesReuse(self):
        """
        Test that prefixes passed to serialization are not modified.

        This test makes sure that passing a dictionary of prefixes repeatedly
        to C{toXml} of elements does not cause serialization errors. A
        previous implementation changed the passed in dictionary internally,
        causing havoc later on.
        """
        prefixes = {'testns': 'foo'}
        s = domish.SerializerClass(prefixes=prefixes)
        self.assertNotIdentical(prefixes, s.prefixes)
        e = domish.Element(('testns2', 'foo'), localPrefixes={'quux': 'testns2'})
        self.assertEqual("<quux:foo xmlns:quux='testns2'/>", e.toXml(prefixes=prefixes))
        e = domish.Element(('testns2', 'foo'))
        self.assertEqual("<foo xmlns='testns2'/>", e.toXml(prefixes=prefixes))

    def testRawXMLSerialization(self):
        e = domish.Element((None, 'foo'))
        e.addRawXml('<abc123>')
        self.assertEqual(e.toXml(), '<foo><abc123></foo>')

    def testRawXMLWithUnicodeSerialization(self):
        e = domish.Element((None, 'foo'))
        e.addRawXml('<degree>°</degree>')
        self.assertEqual(e.toXml(), '<foo><degree>°</degree></foo>')

    def testUnicodeSerialization(self):
        e = domish.Element((None, 'foo'))
        e['test'] = 'my valueȡe'
        e.addContent('A degree symbol...°')
        self.assertEqual(e.toXml(), "<foo test='my valueȡe'>A degree symbol...°</foo>")