import typing as t
from . import nodes
from .visitor import NodeVisitor
def _simple_visit(self, node: nodes.Node, **kwargs: t.Any) -> None:
    for child in node.iter_child_nodes():
        self.sym_visitor.visit(child)