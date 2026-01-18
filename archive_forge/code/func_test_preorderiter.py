from __future__ import annotations
from collections.abc import Iterator
from typing import cast
import pytest
from xarray.core.treenode import InvalidTreeError, NamedNode, NodePath, TreeNode
from xarray.datatree_.datatree.iterators import LevelOrderIter, PreOrderIter
def test_preorderiter(self):
    root, _ = create_test_tree()
    result: list[str | None] = [node.name for node in cast(Iterator[NamedNode], PreOrderIter(root))]
    expected = ['a', 'b', 'd', 'e', 'f', 'g', 'c', 'h', 'i']
    assert result == expected