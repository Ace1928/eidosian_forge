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
def _read_parameter_vec(file_obj, vectors):
    data = formats.PARAMETER_VECTOR_ELEMENT(*struct.unpack(formats.PARAMETER_VECTOR_ELEMENT_PACK, file_obj.read(formats.PARAMETER_VECTOR_ELEMENT_SIZE)))
    param_uuid = uuid.UUID(bytes=data.uuid)
    name = file_obj.read(data.vector_name_size).decode(common.ENCODE)
    if name not in vectors:
        vectors[name] = (ParameterVector(name, data.vector_size), set())
    vector = vectors[name][0]
    if vector[data.index].uuid != param_uuid:
        vectors[name][1].add(data.index)
        vector._params[data.index] = ParameterVectorElement(vector, data.index, uuid=param_uuid)
    return vector[data.index]