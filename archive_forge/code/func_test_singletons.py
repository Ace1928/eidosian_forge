from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_singletons(self) -> None:
    s = '<foo><b/><b /><b\n/></foo>'
    s2 = '<foo><b/><b/><b/></foo>'
    nodes = microdom.parseString(s).documentElement.childNodes
    nodes2 = microdom.parseString(s2).documentElement.childNodes
    self.assertEqual(len(nodes), 3)
    for n, n2 in zip(nodes, nodes2):
        self.assertTrue(isinstance(n, microdom.Element))
        self.assertEqual(n.nodeName, 'b')
        self.assertTrue(n.isEqualToNode(n2))