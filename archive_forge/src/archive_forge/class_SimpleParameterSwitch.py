import pytest, sympy
import cirq
from cirq.study import ParamResolver
class SimpleParameterSwitch:

    def __init__(self, var):
        self.parameter = var

    def _is_parameterized_(self) -> bool:
        return self.parameter != 0

    def _resolve_parameters_(self, resolver: ParamResolver, recursive: bool):
        self.parameter = resolver.value_of(self.parameter, recursive)
        return self