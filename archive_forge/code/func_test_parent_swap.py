from __future__ import annotations
from collections.abc import Iterator
from typing import cast
import pytest
from xarray.core.treenode import InvalidTreeError, NamedNode, NodePath, TreeNode
from xarray.datatree_.datatree.iterators import LevelOrderIter, PreOrderIter
def test_parent_swap(self):
    john: TreeNode = TreeNode()
    mary: TreeNode = TreeNode()
    mary._set_parent(john, 'Mary')
    steve: TreeNode = TreeNode()
    mary._set_parent(steve, 'Mary')
    assert mary.parent == steve
    assert steve.children['Mary'] is mary
    assert 'Mary' not in john.children