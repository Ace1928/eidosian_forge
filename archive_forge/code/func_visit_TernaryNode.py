from __future__ import annotations
import typing as T
def visit_TernaryNode(self, node: mparser.TernaryNode) -> None:
    self.visit_default_func(node)
    node.condition.accept(self)
    node.trueblock.accept(self)
    node.falseblock.accept(self)