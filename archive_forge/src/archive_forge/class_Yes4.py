import numpy as np
import pytest
import cirq
class Yes4(EmptyOp):

    def _has_unitary_(self):
        return NotImplemented

    def _decompose_(self):
        return NotImplemented

    def _apply_unitary_(self, args):
        return NotImplemented

    def _unitary_(self):
        return np.array([[1]])