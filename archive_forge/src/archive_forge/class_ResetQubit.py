import collections
import json
from numbers import Complex
from typing import (
import numpy as np
from deprecated import deprecated
from deprecated.sphinx import versionadded
from pyquil.quilatom import (
from dataclasses import dataclass
class ResetQubit(AbstractInstruction):
    """
    This is the pyQuil object for a Quil targeted reset instruction.
    """

    def __init__(self, qubit: Union[Qubit, QubitPlaceholder, FormalArgument]):
        if not isinstance(qubit, (Qubit, QubitPlaceholder, FormalArgument)):
            raise TypeError('qubit should be a Qubit')
        self.qubit = qubit

    def out(self) -> str:
        return 'RESET {}'.format(self.qubit.out())

    def __str__(self) -> str:
        return 'RESET {}'.format(_format_qubit_str(self.qubit))

    def get_qubits(self, indices: bool=True) -> Set[QubitDesignator]:
        return {_extract_qubit_index(self.qubit, indices)}