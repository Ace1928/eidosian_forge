from __future__ import annotations
import typing as T
def visit_SymbolNode(self, node: mparser.SymbolNode) -> None:
    self.visit_default_func(node)