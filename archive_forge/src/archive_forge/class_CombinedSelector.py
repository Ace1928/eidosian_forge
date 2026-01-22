import sys
import re
import operator
import typing
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union
class CombinedSelector:

    def __init__(self, selector: Tree, combinator: str, subselector: Tree) -> None:
        assert selector is not None
        self.selector = selector
        self.combinator = combinator
        self.subselector = subselector

    def __repr__(self) -> str:
        if self.combinator == ' ':
            comb = '<followed>'
        else:
            comb = self.combinator
        return '%s[%r %s %r]' % (self.__class__.__name__, self.selector, comb, self.subselector)

    def canonical(self) -> str:
        subsel = self.subselector.canonical()
        if len(subsel) > 1:
            subsel = subsel.lstrip('*')
        return '%s %s %s' % (self.selector.canonical(), self.combinator, subsel)

    def specificity(self) -> Tuple[int, int, int]:
        a1, b1, c1 = self.selector.specificity()
        a2, b2, c2 = self.subselector.specificity()
        return (a1 + a2, b1 + b2, c1 + c2)