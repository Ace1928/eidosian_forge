from twisted.trial import unittest
from twisted.words.xish import xpath
from twisted.words.xish.domish import Element
from twisted.words.xish.xpath import XPathQuery
from twisted.words.xish.xpathparser import SyntaxError  # type: ignore[attr-defined]
def test_queryForString(self) -> None:
    """
        queryforString on absolute paths returns their first CDATA.
        """
    xp = XPathQuery('/foo')
    self.assertEqual(xp.queryForString(self.e), 'somecontent')