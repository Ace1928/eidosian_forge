from __future__ import annotations
import typing as T
def visit_MultilineStringNode(self, node: mparser.MultilineFormatStringNode) -> None:
    self.visit_default_func(node)