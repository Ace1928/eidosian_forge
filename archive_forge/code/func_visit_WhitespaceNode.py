from __future__ import annotations
import typing as T
def visit_WhitespaceNode(self, node: mparser.WhitespaceNode) -> None:
    self.visit_default_func(node)