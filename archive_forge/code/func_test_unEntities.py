from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_unEntities(self) -> None:
    s = '\n                <HTML>\n                    This HTML goes between Stupid <=CrAzY!=> Dumb.\n                </HTML>\n            '
    d = microdom.parseString(s, beExtremelyLenient=1)
    n = domhelpers.gatherTextNodes(d)
    self.assertNotEqual(n.find('>'), -1)