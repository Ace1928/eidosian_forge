from __future__ import annotations
from collections.abc import Iterator
from typing import cast
import pytest
from xarray.core.treenode import InvalidTreeError, NamedNode, NodePath, TreeNode
from xarray.datatree_.datatree.iterators import LevelOrderIter, PreOrderIter
def test_subtree(self):
    root, _ = create_test_tree()
    subtree = root.subtree
    expected = ['a', 'b', 'd', 'e', 'f', 'g', 'c', 'h', 'i']
    for node, expected_name in zip(subtree, expected):
        assert node.name == expected_name