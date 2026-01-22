import sys
import re
import operator
import typing
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union
class SpecificityAdjustment:
    """
    Represents selector:where(selector_list)
    Same as selector:is(selector_list), but its specificity is always 0
    """

    def __init__(self, selector: Tree, selector_list: List[Tree]):
        self.selector = selector
        self.selector_list = selector_list

    def __repr__(self) -> str:
        return '%s[%r:where(%s)]' % (self.__class__.__name__, self.selector, ', '.join(map(repr, self.selector_list)))

    def canonical(self) -> str:
        selector_arguments = []
        for s in self.selector_list:
            selarg = s.canonical()
            selector_arguments.append(selarg.lstrip('*'))
        return '%s:where(%s)' % (self.selector.canonical(), ', '.join(map(str, selector_arguments)))

    def specificity(self) -> Tuple[int, int, int]:
        return (0, 0, 0)