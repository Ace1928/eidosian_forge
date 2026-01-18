from __future__ import annotations
import typing as T
def visit_IfNode(self, node: mparser.IfNode) -> None:
    self.visit_default_func(node)
    node.condition.accept(self)
    node.block.accept(self)