from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
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