from __future__ import annotations
from .. import mparser
from .visitor import AstVisitor
from itertools import zip_longest
import re
import typing as T
def append_padded(self, data: str, node: mparser.BaseNode) -> None:
    if self.result and self.result[-1] not in [' ', '\n']:
        data = ' ' + data
    self.append(data + ' ', node)