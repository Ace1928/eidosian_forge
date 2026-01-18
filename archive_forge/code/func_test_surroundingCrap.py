from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_surroundingCrap(self) -> None:
    """
        If a document is surrounded by non-xml text, the text should
        be remain in the XML.
        """
    s = 'Hi<br> orders!'
    d = microdom.parseString(s, beExtremelyLenient=True)
    self.assertEqual(d.firstChild().toxml(), '<html>Hi<br /> orders!</html>')