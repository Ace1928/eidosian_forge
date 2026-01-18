from twisted.trial import unittest
from twisted.words.xish import xpath
from twisted.words.xish.domish import Element
from twisted.words.xish.xpath import XPathQuery
from twisted.words.xish.xpathparser import SyntaxError  # type: ignore[attr-defined]
def test_orOperator(self) -> None:
    """
        Test boolean or operator in condition.
        """
    xp = XPathQuery("//bar[@attrib5='value4' or @attrib5='value5']")
    self.assertEqual(xp.matches(self.e), True)
    self.assertEqual(xp.queryForNodes(self.e), [self.bar5, self.bar6])