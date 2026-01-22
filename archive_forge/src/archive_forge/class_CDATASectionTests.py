from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
class CDATASectionTests(TestCase):
    """
    Tests for L{CDATASection}.
    """

    def test_isEqualToNode(self) -> None:
        """
        L{CDATASection.isEqualToNode} returns C{True} if and only if passed a
        L{CDATASection} which represents the same data.
        """
        self.assertTrue(microdom.CDATASection('foo').isEqualToNode(microdom.CDATASection('foo')))
        self.assertFalse(microdom.CDATASection('foo').isEqualToNode(microdom.CDATASection('bar')))