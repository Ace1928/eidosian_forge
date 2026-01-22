from typing import TypeVar, Tuple, List, Callable, Generic, Type, Union, Optional, Any, cast
from abc import ABC
from .utils import combine_alternatives
from .tree import Tree, Branch
from .exceptions import VisitError, GrammarError
from .lexer import Token
from functools import wraps, update_wrapper
from inspect import getmembers, getmro
class TransformerChain(Generic[_Leaf_T, _Return_T]):
    transformers: 'Tuple[Union[Transformer, TransformerChain], ...]'

    def __init__(self, *transformers: 'Union[Transformer, TransformerChain]') -> None:
        self.transformers = transformers

    def transform(self, tree: Tree[_Leaf_T]) -> _Return_T:
        for t in self.transformers:
            tree = t.transform(tree)
        return cast(_Return_T, tree)

    def __mul__(self: 'TransformerChain[_Leaf_T, Tree[_Leaf_U]]', other: 'Union[Transformer[_Leaf_U, _Return_V], TransformerChain[_Leaf_U, _Return_V]]') -> 'TransformerChain[_Leaf_T, _Return_V]':
        return TransformerChain(*self.transformers + (other,))