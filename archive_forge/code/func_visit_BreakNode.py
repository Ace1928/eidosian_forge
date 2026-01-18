from __future__ import annotations
import typing as T
def visit_BreakNode(self, node: mparser.BreakNode) -> None:
    self.visit_default_func(node)