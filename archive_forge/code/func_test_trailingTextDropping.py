from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_trailingTextDropping(self) -> None:
    """
        Ensure that no *trailing* text in a mal-formed
        no-top-level-element document(s) will not be dropped.
        """
    s = '<br>Hi orders!'
    d = microdom.parseString(s, beExtremelyLenient=True)
    self.assertEqual(d.firstChild().toxml(), '<html><br />Hi orders!</html>')
    byteStream = BytesIO()
    d.firstChild().writexml(byteStream, '', '', '', '', {}, '')
    self.assertEqual(byteStream.getvalue(), b'<html><br />Hi orders!</html>')