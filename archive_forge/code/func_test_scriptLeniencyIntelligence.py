from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_scriptLeniencyIntelligence(self) -> None:
    s = '<script><!-- lalal --></script>'
    self.assertEqual(microdom.parseString(s, beExtremelyLenient=1).firstChild().toxml(), s)
    s = '<script><![CDATA[lalal]]></script>'
    self.assertEqual(microdom.parseString(s, beExtremelyLenient=1).firstChild().toxml(), s)
    s = '<script> // <![CDATA[\n        lalal\n        //]]></script>'
    self.assertEqual(microdom.parseString(s, beExtremelyLenient=1).firstChild().toxml(), s)