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
def _read_parameter(file_obj):
    data = formats.PARAMETER(*struct.unpack(formats.PARAMETER_PACK, file_obj.read(formats.PARAMETER_SIZE)))
    param_uuid = uuid.UUID(bytes=data.uuid)
    name = file_obj.read(data.name_size).decode(common.ENCODE)
    return Parameter(name, uuid=param_uuid)