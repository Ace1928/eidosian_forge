from __future__ import annotations
from .. import mparser
from .visitor import AstVisitor
from itertools import zip_longest
import re
import typing as T
def _accept_list(self, key: str, nodes: T.Sequence[mparser.BaseNode]) -> None:
    old = self.current
    datalist: T.List[T.Dict[str, T.Any]] = []
    for i in nodes:
        self.current = {}
        i.accept(self)
        datalist += [self.current]
    self.current = old
    self.current[key] = datalist