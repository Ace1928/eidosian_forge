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
def _read_expr_type(file_obj) -> types.Type:
    type_key = file_obj.read(formats.EXPR_TYPE_DISCRIMINATOR_SIZE)
    if type_key == type_keys.ExprType.BOOL:
        return types.Bool()
    if type_key == type_keys.ExprType.UINT:
        elem = formats.EXPR_TYPE_UINT._make(struct.unpack(formats.EXPR_TYPE_UINT_PACK, file_obj.read(formats.EXPR_TYPE_UINT_SIZE)))
        return types.Uint(elem.width)
    raise exceptions.QpyError(f"Invalid classical-expression Type key '{type_key}'")