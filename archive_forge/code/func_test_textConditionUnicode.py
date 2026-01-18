from twisted.trial import unittest
from twisted.words.xish import xpath
from twisted.words.xish.domish import Element
from twisted.words.xish.xpath import XPathQuery
from twisted.words.xish.xpathparser import SyntaxError  # type: ignore[attr-defined]
def test_textConditionUnicode(self) -> None:
    """
        A node can be matched by text with non-ascii code points.
        """
    xp = XPathQuery("//*[text()='☃']")
    self.assertEqual(xp.matches(self.e), True)
    self.assertEqual(xp.queryForNodes(self.e), [self.quux])