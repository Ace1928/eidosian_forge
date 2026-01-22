import collections
import json
from numbers import Complex
from typing import (
import numpy as np
from deprecated import deprecated
from deprecated.sphinx import versionadded
from pyquil.quilatom import (
from dataclasses import dataclass
class DefGateByPaulis(DefGate):
    """
    Records a gate definition as the exponentiation of a PauliSum.
    """

    def __init__(self, gate_name: str, parameters: List[Parameter], arguments: List[QubitDesignator], body: 'PauliSum'):
        if not isinstance(gate_name, str):
            raise TypeError('Gate name must be a string')
        if gate_name in RESERVED_WORDS:
            raise ValueError(f"Cannot use {gate_name} for a gate name since it's a reserved word")
        self.name = gate_name
        self.parameters = parameters
        self.arguments = arguments
        self.body = body

    def out(self) -> str:
        out = f'DEFGATE {self.name}'
        if self.parameters is not None:
            out += f'({', '.join(map(str, self.parameters))}) '
        out += f'{' '.join(map(str, self.arguments))} AS PAULI-SUM:\n'
        for term in self.body:
            args = term._ops.keys()
            word = term._ops.values()
            out += f'    {''.join(word)}({term.coefficient}) ' + ' '.join(map(str, args)) + '\n'
        return out

    def num_args(self) -> int:
        return len(self.arguments)