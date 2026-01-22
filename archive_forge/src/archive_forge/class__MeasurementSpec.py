import dataclasses
import numbers
from typing import (
import sympy
from cirq import ops, value, protocols
@dataclasses.dataclass(frozen=True)
class _MeasurementSpec:
    """An encapsulation of all the specifications for one run of a
    quantum processor.

    This includes the maximal input-output setting (which may result in many
    observables being measured if they are consistent with `max_setting`) and
    a set of circuit parameters if the circuit is parameterized.
    """
    max_setting: InitObsSetting
    circuit_params: Mapping[Union[str, sympy.Expr], Union[value.Scalar, sympy.Expr]]

    def __hash__(self):
        return hash((self.max_setting, _hashable_param(self.circuit_params.items())))

    def __repr__(self):
        return f'cirq.work._MeasurementSpec(max_setting={self.max_setting!r}, circuit_params={self.circuit_params!r})'

    def _json_dict_(self):
        return protocols.dataclass_json_dict(self)