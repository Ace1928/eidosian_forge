import numpy as np
import pytest
import cirq
def test_via_has_unitary():

    class No1:

        def _has_unitary_(self):
            return NotImplemented

    class No2:

        def _has_unitary_(self):
            return False

    class Yes:

        def _has_unitary_(self):
            return True
    assert not cirq.has_unitary(No1())
    assert not cirq.has_unitary(No2())
    assert cirq.has_unitary(Yes())