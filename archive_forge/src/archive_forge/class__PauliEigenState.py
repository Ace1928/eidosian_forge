import abc
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
from cirq import protocols
from cirq._doc import document
class _PauliEigenState(_NamedOneQubitState):

    def __init__(self, eigenvalue: int):
        self.eigenvalue = eigenvalue
        self._eigen_index = (1 - eigenvalue) / 2

    @property
    @abc.abstractmethod
    def _symbol(self) -> str:
        pass

    def __str__(self) -> str:
        sign = {1: '+', -1: '-'}[self.eigenvalue]
        return f'{sign}{self._symbol}'

    def __repr__(self) -> str:
        return f'cirq.{self._symbol}.basis[{self.eigenvalue:+d}]'

    @abc.abstractmethod
    def stabilized_by(self) -> Tuple[int, 'cirq.Pauli']:
        pass

    def __eq__(self, other) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.eigenvalue == other.eigenvalue

    def __hash__(self):
        return hash((self.__class__.__name__, self.eigenvalue))

    def _json_dict_(self):
        return protocols.obj_to_dict_helper(self, ['eigenvalue'])