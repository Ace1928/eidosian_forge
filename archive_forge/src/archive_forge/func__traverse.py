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
def _traverse(self, predicate: Callable[[Union['Traversable', 'Blob', TraversedTup], int], bool]=lambda i, d: True, prune: Callable[[Union['Traversable', 'Blob', TraversedTup], int], bool]=lambda i, d: False, depth: int=-1, branch_first: bool=True, visit_once: bool=True, ignore_self: int=1, as_edge: bool=False) -> Union[Iterator[Union['Traversable', 'Blob']], Iterator[TraversedTup]]:
    """Iterator yielding items found when traversing self.

        :param predicate: f(i,d) returns False if item i at depth d should not be
            included in the result.

        :param prune:
            f(i,d) return True if the search should stop at item i at depth d. Item i
            will not be returned.

        :param depth:
            Defines at which level the iteration should not go deeper if -1, there is no
            limit if 0, you would effectively only get self, the root of the iteration
            i.e. if 1, you would only get the first level of predecessors/successors

        :param branch_first:
            if True, items will be returned branch first, otherwise depth first

        :param visit_once:
            if True, items will only be returned once, although they might be
            encountered several times. Loops are prevented that way.

        :param ignore_self:
            if True, self will be ignored and automatically pruned from the result.
            Otherwise it will be the first item to be returned. If as_edge is True, the
            source of the first edge is None

        :param as_edge:
            if True, return a pair of items, first being the source, second the
            destination, i.e. tuple(src, dest) with the edge spanning from source to
            destination

        :return: Iterator yielding items found when traversing self::

                Commit -> Iterator[Union[Commit, Tuple[Commit, Commit]]
                Submodule -> Iterator[Submodule, Tuple[Submodule, Submodule]]
                Tree -> Iterator[Union[Blob, Tree, Submodule,
                                        Tuple[Union[Submodule, Tree], Union[Blob, Tree, Submodule]]]

                ignore_self=True is_edge=True -> Iterator[item]
                ignore_self=True is_edge=False --> Iterator[item]
                ignore_self=False is_edge=True -> Iterator[item] | Iterator[Tuple[src, item]]
                ignore_self=False is_edge=False -> Iterator[Tuple[src, item]]
        """
    visited = set()
    stack: Deque[TraverseNT] = deque()
    stack.append(TraverseNT(0, self, None))

    def addToStack(stack: Deque[TraverseNT], src_item: 'Traversable', branch_first: bool, depth: int) -> None:
        lst = self._get_intermediate_items(item)
        if not lst:
            return
        if branch_first:
            stack.extendleft((TraverseNT(depth, i, src_item) for i in lst))
        else:
            reviter = (TraverseNT(depth, lst[i], src_item) for i in range(len(lst) - 1, -1, -1))
            stack.extend(reviter)
    while stack:
        d, item, src = stack.pop()
        if visit_once and item in visited:
            continue
        if visit_once:
            visited.add(item)
        rval: Union[TraversedTup, 'Traversable', 'Blob']
        if as_edge:
            rval = (src, item)
        else:
            rval = item
        if prune(rval, d):
            continue
        skipStartItem = ignore_self and item is self
        if not skipStartItem and predicate(rval, d):
            yield rval
        nd = d + 1
        if depth > -1 and nd > depth:
            continue
        addToStack(stack, item, branch_first, nd)