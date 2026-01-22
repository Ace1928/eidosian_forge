from __future__ import annotations
from collections.abc import Iterator
from typing import cast
import pytest
from xarray.core.treenode import InvalidTreeError, NamedNode, NodePath, TreeNode
from xarray.datatree_.datatree.iterators import LevelOrderIter, PreOrderIter
class TestSetNodes:

    def test_set_child_node(self):
        john: TreeNode = TreeNode()
        mary: TreeNode = TreeNode()
        john._set_item('Mary', mary)
        assert john.children['Mary'] is mary
        assert isinstance(mary, TreeNode)
        assert mary.children == {}
        assert mary.parent is john

    def test_child_already_exists(self):
        mary: TreeNode = TreeNode()
        john: TreeNode = TreeNode(children={'Mary': mary})
        mary_2: TreeNode = TreeNode()
        with pytest.raises(KeyError):
            john._set_item('Mary', mary_2, allow_overwrite=False)

    def test_set_grandchild(self):
        rose: TreeNode = TreeNode()
        mary: TreeNode = TreeNode()
        john: TreeNode = TreeNode()
        john._set_item('Mary', mary)
        john._set_item('Mary/Rose', rose)
        assert john.children['Mary'] is mary
        assert isinstance(mary, TreeNode)
        assert 'Rose' in mary.children
        assert rose.parent is mary

    def test_create_intermediate_child(self):
        john: TreeNode = TreeNode()
        rose: TreeNode = TreeNode()
        with pytest.raises(KeyError, match='Could not reach'):
            john._set_item(path='Mary/Rose', item=rose, new_nodes_along_path=False)
        john._set_item('Mary/Rose', rose, new_nodes_along_path=True)
        assert 'Mary' in john.children
        mary = john.children['Mary']
        assert isinstance(mary, TreeNode)
        assert mary.children == {'Rose': rose}
        assert rose.parent == mary
        assert rose.parent == mary

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