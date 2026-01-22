import numpy as np
import pytest
import cirq
class Yes1(EmptyOp):

    def _has_unitary_(self):
        return True

    def _decompose_(self):
        assert False

    def _apply_unitary_(self, args):
        assert False

    def _unitary_(self):
        assert False