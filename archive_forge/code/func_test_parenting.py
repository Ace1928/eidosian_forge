from __future__ import annotations
from collections.abc import Iterator
from typing import cast
import pytest
from xarray.core.treenode import InvalidTreeError, NamedNode, NodePath, TreeNode
from xarray.datatree_.datatree.iterators import LevelOrderIter, PreOrderIter
def test_parenting(self):
    john: TreeNode = TreeNode()
    mary: TreeNode = TreeNode()
    mary._set_parent(john, 'Mary')
    assert mary.parent == john
    assert john.children['Mary'] is mary