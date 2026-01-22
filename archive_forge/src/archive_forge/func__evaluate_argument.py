import dataclasses
import math
from typing import Iterable, Callable
from qiskit.circuit import (
from qiskit._qasm2 import (  # pylint: disable=no-name-in-module
from .exceptions import QASM2ParseError
def _evaluate_argument(expr, parameters):
    """Inner recursive function to calculate the value of a mathematical expression given the
    concrete values in the `parameters` field."""
    if isinstance(expr, ExprConstant):
        return expr.value
    if isinstance(expr, ExprArgument):
        return parameters[expr.index]
    if isinstance(expr, ExprUnary):
        inner = _evaluate_argument(expr.argument, parameters)
        opcode = expr.opcode
        if opcode == UnaryOpCode.Negate:
            return -inner
        if opcode == UnaryOpCode.Cos:
            return math.cos(inner)
        if opcode == UnaryOpCode.Exp:
            return math.exp(inner)
        if opcode == UnaryOpCode.Ln:
            return math.log(inner)
        if opcode == UnaryOpCode.Sin:
            return math.sin(inner)
        if opcode == UnaryOpCode.Sqrt:
            return math.sqrt(inner)
        if opcode == UnaryOpCode.Tan:
            return math.tan(inner)
        raise ValueError(f'unhandled unary opcode: {opcode}')
    if isinstance(expr, ExprBinary):
        left = _evaluate_argument(expr.left, parameters)
        right = _evaluate_argument(expr.right, parameters)
        opcode = expr.opcode
        if opcode == BinaryOpCode.Add:
            return left + right
        if opcode == BinaryOpCode.Subtract:
            return left - right
        if opcode == BinaryOpCode.Multiply:
            return left * right
        if opcode == BinaryOpCode.Divide:
            return left / right
        if opcode == BinaryOpCode.Power:
            return left ** right
        raise ValueError(f'unhandled binary opcode: {opcode}')
    if isinstance(expr, ExprCustom):
        return expr.callable(*(_evaluate_argument(x, parameters) for x in expr.arguments))
    raise ValueError(f'unhandled expression type: {expr}')