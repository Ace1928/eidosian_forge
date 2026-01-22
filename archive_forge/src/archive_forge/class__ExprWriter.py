from __future__ import annotations
import collections.abc
import struct
import uuid
import numpy as np
import symengine
from symengine.lib.symengine_wrapper import (  # pylint: disable = no-name-in-module
from qiskit.circuit import CASE_DEFAULT, Clbit, ClassicalRegister
from qiskit.circuit.classical import expr, types
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.parametervector import ParameterVector, ParameterVectorElement
from qiskit.qpy import common, formats, exceptions, type_keys
class _ExprWriter(expr.ExprVisitor[None]):
    __slots__ = ('file_obj', 'clbit_indices')

    def __init__(self, file_obj, clbit_indices):
        self.file_obj = file_obj
        self.clbit_indices = clbit_indices

    def visit_generic(self, node, /):
        raise exceptions.QpyError(f"unhandled Expr object '{node}'")

    def visit_var(self, node, /):
        self.file_obj.write(type_keys.Expression.VAR)
        _write_expr_type(self.file_obj, node.type)
        if isinstance(node.var, Clbit):
            self.file_obj.write(type_keys.ExprVar.CLBIT)
            self.file_obj.write(struct.pack(formats.EXPR_VAR_CLBIT_PACK, *formats.EXPR_VAR_CLBIT(self.clbit_indices[node.var])))
        elif isinstance(node.var, ClassicalRegister):
            self.file_obj.write(type_keys.ExprVar.REGISTER)
            self.file_obj.write(struct.pack(formats.EXPR_VAR_REGISTER_PACK, *formats.EXPR_VAR_REGISTER(len(node.var.name))))
            self.file_obj.write(node.var.name.encode(common.ENCODE))
        else:
            raise exceptions.QpyError(f"unhandled Var object '{node.var}'")

    def visit_value(self, node, /):
        self.file_obj.write(type_keys.Expression.VALUE)
        _write_expr_type(self.file_obj, node.type)
        if node.value is True or node.value is False:
            self.file_obj.write(type_keys.ExprValue.BOOL)
            self.file_obj.write(struct.pack(formats.EXPR_VALUE_BOOL_PACK, *formats.EXPR_VALUE_BOOL(node.value)))
        elif isinstance(node.value, int):
            self.file_obj.write(type_keys.ExprValue.INT)
            if node.value == 0:
                num_bytes = 0
                buffer = b''
            else:
                num_bytes = node.value.bit_length() // 8 + 1
                buffer = node.value.to_bytes(num_bytes, 'big', signed=True)
            self.file_obj.write(struct.pack(formats.EXPR_VALUE_INT_PACK, *formats.EXPR_VALUE_INT(num_bytes)))
            self.file_obj.write(buffer)
        else:
            raise exceptions.QpyError(f"unhandled Value object '{node.value}'")

    def visit_cast(self, node, /):
        self.file_obj.write(type_keys.Expression.CAST)
        _write_expr_type(self.file_obj, node.type)
        self.file_obj.write(struct.pack(formats.EXPRESSION_CAST_PACK, *formats.EXPRESSION_CAST(node.implicit)))
        node.operand.accept(self)

    def visit_unary(self, node, /):
        self.file_obj.write(type_keys.Expression.UNARY)
        _write_expr_type(self.file_obj, node.type)
        self.file_obj.write(struct.pack(formats.EXPRESSION_UNARY_PACK, *formats.EXPRESSION_UNARY(node.op.value)))
        node.operand.accept(self)

    def visit_binary(self, node, /):
        self.file_obj.write(type_keys.Expression.BINARY)
        _write_expr_type(self.file_obj, node.type)
        self.file_obj.write(struct.pack(formats.EXPRESSION_BINARY_PACK, *formats.EXPRESSION_UNARY(node.op.value)))
        node.left.accept(self)
        node.right.accept(self)