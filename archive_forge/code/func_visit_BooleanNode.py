from __future__ import annotations
import typing as T
def visit_BooleanNode(self, node: mparser.BooleanNode) -> None:
    self.visit_default_func(node)