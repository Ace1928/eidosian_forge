from __future__ import annotations
import typing as T
def visit_ContinueNode(self, node: mparser.ContinueNode) -> None:
    self.visit_default_func(node)