from twisted.trial import unittest
from twisted.words.xish import xpath
from twisted.words.xish.domish import Element
from twisted.words.xish.xpath import XPathQuery
from twisted.words.xish.xpathparser import SyntaxError  # type: ignore[attr-defined]
def test_queryForStringList(self) -> None:
    """
        queryforStringList on absolute paths returns all their CDATA.
        """
    xp = XPathQuery('/foo')
    self.assertEqual(xp.queryForStringList(self.e), ['somecontent', 'somemorecontent'])