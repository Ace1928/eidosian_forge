from typing import TypeVar, Tuple, List, Callable, Generic, Type, Union, Optional, Any, cast
from abc import ABC
from .utils import combine_alternatives
from .tree import Tree, Branch
from .exceptions import VisitError, GrammarError
from .lexer import Token
from functools import wraps, update_wrapper
from inspect import getmembers, getmro
def visit_topdown(self, tree: Tree[_Leaf_T]) -> Tree[_Leaf_T]:
    """Visit the tree, starting at the root, and ending at the leaves (top-down)"""
    self._call_userfunc(tree)
    for child in tree.children:
        if isinstance(child, Tree):
            self.visit_topdown(child)
    return tree