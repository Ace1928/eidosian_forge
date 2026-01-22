from abc import ABC, abstractmethod
import calendar
from collections import deque
from datetime import datetime, timedelta, tzinfo
from string import digits
import re
import time
import warnings
from git.util import IterableList, IterableObj, Actor
from typing import (
from git.types import Has_id_attribute, Literal  # , _T
class TraversableIterableObj(IterableObj, Traversable):
    __slots__ = ()
    TIobj_tuple = Tuple[Union[T_TIobj, None], T_TIobj]

    def list_traverse(self: T_TIobj, *args: Any, **kwargs: Any) -> IterableList[T_TIobj]:
        return super()._list_traverse(*args, **kwargs)

    @overload
    def traverse(self: T_TIobj) -> Iterator[T_TIobj]:
        ...

    @overload
    def traverse(self: T_TIobj, predicate: Callable[[Union[T_TIobj, Tuple[Union[T_TIobj, None], T_TIobj]], int], bool], prune: Callable[[Union[T_TIobj, Tuple[Union[T_TIobj, None], T_TIobj]], int], bool], depth: int, branch_first: bool, visit_once: bool, ignore_self: Literal[True], as_edge: Literal[False]) -> Iterator[T_TIobj]:
        ...

    @overload
    def traverse(self: T_TIobj, predicate: Callable[[Union[T_TIobj, Tuple[Union[T_TIobj, None], T_TIobj]], int], bool], prune: Callable[[Union[T_TIobj, Tuple[Union[T_TIobj, None], T_TIobj]], int], bool], depth: int, branch_first: bool, visit_once: bool, ignore_self: Literal[False], as_edge: Literal[True]) -> Iterator[Tuple[Union[T_TIobj, None], T_TIobj]]:
        ...

    @overload
    def traverse(self: T_TIobj, predicate: Callable[[Union[T_TIobj, TIobj_tuple], int], bool], prune: Callable[[Union[T_TIobj, TIobj_tuple], int], bool], depth: int, branch_first: bool, visit_once: bool, ignore_self: Literal[True], as_edge: Literal[True]) -> Iterator[Tuple[T_TIobj, T_TIobj]]:
        ...

    def traverse(self: T_TIobj, predicate: Callable[[Union[T_TIobj, TIobj_tuple], int], bool]=lambda i, d: True, prune: Callable[[Union[T_TIobj, TIobj_tuple], int], bool]=lambda i, d: False, depth: int=-1, branch_first: bool=True, visit_once: bool=True, ignore_self: int=1, as_edge: bool=False) -> Union[Iterator[T_TIobj], Iterator[Tuple[T_TIobj, T_TIobj]], Iterator[TIobj_tuple]]:
        """For documentation, see :meth:`Traversable._traverse`."""
        return cast(Union[Iterator[T_TIobj], Iterator[Tuple[Union[None, T_TIobj], T_TIobj]]], super()._traverse(predicate, prune, depth, branch_first, visit_once, ignore_self, as_edge))