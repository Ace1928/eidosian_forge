from __future__ import annotations
import typing as T
def visit_DictNode(self, node: mparser.DictNode) -> None:
    self.visit_default_func(node)
    node.args.accept(self)