import numpy as np
import pytest
import cirq
def test_via_apply_unitary():

    class No1(EmptyOp):

        def _apply_unitary_(self, args):
            return None

    class No2(EmptyOp):

        def _apply_unitary_(self, args):
            return NotImplemented

    class No3(cirq.testing.SingleQubitGate):

        def _apply_unitary_(self, args):
            return NotImplemented

    class No4:

        def _apply_unitary_(self, args):
            assert False

    class Yes1(EmptyOp):

        def _apply_unitary_(self, args):
            return args.target_tensor

    class Yes2(cirq.testing.SingleQubitGate):

        def _apply_unitary_(self, args):
            return args.target_tensor
    assert cirq.has_unitary(Yes1())
    assert cirq.has_unitary(Yes1(), allow_decompose=False)
    assert cirq.has_unitary(Yes2())
    assert not cirq.has_unitary(No1())
    assert not cirq.has_unitary(No2())
    assert not cirq.has_unitary(No3())
    assert not cirq.has_unitary(No4())