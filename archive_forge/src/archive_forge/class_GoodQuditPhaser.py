import pytest
import numpy as np
import cirq
class GoodQuditPhaser:

    def __init__(self, e):
        self.e = e

    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.array([[0, 1j ** (-self.e), 0], [0, 0, 1j ** self.e], [1, 0, 0]])

    def _phase_by_(self, phase_turns: float, qubit_index: int):
        return GoodQuditPhaser(self.e + phase_turns * 4)

    def _resolve_parameters_(self, resolver, recursive):
        return GoodQuditPhaser(resolver.value_of(self.e, recursive))