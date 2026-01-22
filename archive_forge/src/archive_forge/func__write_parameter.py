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
def _write_parameter(file_obj, obj):
    name_bytes = obj.name.encode(common.ENCODE)
    file_obj.write(struct.pack(formats.PARAMETER_PACK, len(name_bytes), obj.uuid.bytes))
    file_obj.write(name_bytes)