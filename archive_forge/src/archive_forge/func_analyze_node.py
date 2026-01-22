import typing as t
from . import nodes
from .visitor import NodeVisitor
def analyze_node(self, node: nodes.Node, **kwargs: t.Any) -> None:
    visitor = RootVisitor(self)
    visitor.visit(node, **kwargs)