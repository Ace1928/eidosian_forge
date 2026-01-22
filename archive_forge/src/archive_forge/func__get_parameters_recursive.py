from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.pulse.channels import Channel
from qiskit.pulse.exceptions import PulseError
def _get_parameters_recursive(obj):
    params = set()
    if hasattr(obj, 'parameters'):
        for param in obj.parameters:
            if isinstance(param, Parameter):
                params.add(param)
            else:
                params |= _get_parameters_recursive(param)
    return params