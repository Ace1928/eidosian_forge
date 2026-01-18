from twisted.trial import unittest
from twisted.words.xish import xpath
from twisted.words.xish.domish import Element
from twisted.words.xish.xpath import XPathQuery
from twisted.words.xish.xpathparser import SyntaxError  # type: ignore[attr-defined]
def test_staticMethods(self) -> None:
    """
        Test basic operation of the static methods.
        """
    self.assertEqual(xpath.matches('/foo/bar', self.e), True)
    self.assertEqual(xpath.queryForNodes('/foo/bar', self.e), [self.bar1, self.bar2, self.bar4, self.bar5, self.bar6, self.bar7])
    self.assertEqual(xpath.queryForString('/foo', self.e), 'somecontent')
    self.assertEqual(xpath.queryForStringList('/foo', self.e), ['somecontent', 'somemorecontent'])