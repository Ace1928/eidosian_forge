from __future__ import annotations
from collections.abc import Iterator
from typing import cast
import pytest
from xarray.core.treenode import InvalidTreeError, NamedNode, NodePath, TreeNode
from xarray.datatree_.datatree.iterators import LevelOrderIter, PreOrderIter
def test_overwrite_child(self):
    john: TreeNode = TreeNode()
    mary: TreeNode = TreeNode()
    john._set_item('Mary', mary)
    marys_evil_twin: TreeNode = TreeNode()
    with pytest.raises(KeyError, match='Already a node object'):
        john._set_item('Mary', marys_evil_twin, allow_overwrite=False)
    assert john.children['Mary'] is mary
    assert marys_evil_twin.parent is None
    marys_evil_twin = TreeNode()
    john._set_item('Mary', marys_evil_twin, allow_overwrite=True)
    assert john.children['Mary'] is marys_evil_twin
    assert marys_evil_twin.parent is john