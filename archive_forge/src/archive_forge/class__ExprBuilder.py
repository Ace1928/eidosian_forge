import collections
import re
import io
import itertools
import numbers
from os.path import dirname, join, abspath
from typing import Iterable, List, Sequence, Union
from qiskit.circuit import (
from qiskit.circuit.bit import Bit
from qiskit.circuit.classical import expr, types
from qiskit.circuit.controlflow import (
from qiskit.circuit.library import standard_gates
from qiskit.circuit.register import Register
from qiskit.circuit.tools import pi_check
from . import ast
from .experimental import ExperimentalFeatures
from .exceptions import QASM3ExporterError
from .printer import BasicPrinter
class _ExprBuilder(expr.ExprVisitor[ast.Expression]):
    __slots__ = ('lookup',)

    def __init__(self, lookup):
        self.lookup = lookup

    def visit_var(self, node, /):
        return self.lookup(node.var)

    def visit_value(self, node, /):
        if node.type.kind is types.Bool:
            return ast.BooleanLiteral(node.value)
        if node.type.kind is types.Uint:
            return ast.IntegerLiteral(node.value)
        raise RuntimeError(f"unhandled Value type '{node}'")

    def visit_cast(self, node, /):
        if node.implicit:
            return node.accept(self)
        if node.type.kind is types.Bool:
            oq3_type = ast.BoolType()
        elif node.type.kind is types.Uint:
            oq3_type = ast.BitArrayType(node.type.width)
        else:
            raise RuntimeError(f"unhandled cast type '{node.type}'")
        return ast.Cast(oq3_type, node.operand.accept(self))

    def visit_unary(self, node, /):
        return ast.Unary(ast.Unary.Op[node.op.name], node.operand.accept(self))

    def visit_binary(self, node, /):
        return ast.Binary(ast.Binary.Op[node.op.name], node.left.accept(self), node.right.accept(self))