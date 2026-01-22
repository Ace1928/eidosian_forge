from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
class NodeTests(TestCase):
    """
    Tests for L{Node}.
    """

    def test_isNodeEqualTo(self) -> None:
        """
        L{Node.isEqualToNode} returns C{True} if and only if passed a L{Node}
        with the same children.
        """
        node = microdom.Node(object())
        self.assertTrue(node.isEqualToNode(node))
        another = microdom.Node(object())
        self.assertTrue(node.isEqualToNode(another))
        node.appendChild(microdom.Node(object()))
        self.assertFalse(node.isEqualToNode(another))
        another.appendChild(microdom.Node(object()))
        self.assertTrue(node.isEqualToNode(another))
        node.firstChild().appendChild(microdom.Node(object()))
        self.assertFalse(node.isEqualToNode(another))
        another.firstChild().appendChild(microdom.Node(object()))
        self.assertTrue(node.isEqualToNode(another))

    def test_validChildInstance(self) -> None:
        """
        Children of L{Node} instances must also be L{Node} instances.
        """
        node = microdom.Node()
        child = microdom.Node()
        node.appendChild(child)
        self.assertRaises(TypeError, node.appendChild, None)
        self.assertRaises(TypeError, node.insertBefore, child, None)
        self.assertRaises(TypeError, node.insertBefore, None, child)
        self.assertRaises(TypeError, node.insertBefore, None, None)
        node.removeChild(child)
        self.assertRaises(TypeError, node.removeChild, None)
        self.assertRaises(TypeError, node.replaceChild, child, None)
        self.assertRaises(TypeError, node.replaceChild, None, child)
        self.assertRaises(TypeError, node.replaceChild, None, None)