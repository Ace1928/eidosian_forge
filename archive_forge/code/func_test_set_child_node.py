from __future__ import annotations
from collections.abc import Iterator
from typing import cast
import pytest
from xarray.core.treenode import InvalidTreeError, NamedNode, NodePath, TreeNode
from xarray.datatree_.datatree.iterators import LevelOrderIter, PreOrderIter
def test_set_child_node(self):
    john: TreeNode = TreeNode()
    mary: TreeNode = TreeNode()
    john._set_item('Mary', mary)
    assert john.children['Mary'] is mary
    assert isinstance(mary, TreeNode)
    assert mary.children == {}
    assert mary.parent is john