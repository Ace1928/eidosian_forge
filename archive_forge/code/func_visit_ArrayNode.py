from __future__ import annotations
import typing as T
def visit_ArrayNode(self, node: mparser.ArrayNode) -> None:
    self.visit_default_func(node)
    node.args.accept(self)