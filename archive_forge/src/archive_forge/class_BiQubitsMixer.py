from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union
import attr
import cirq
from cirq._compat import cached_property
from numpy.typing import NDArray
from cirq_ft import infra
from cirq_ft.algos import and_gate
from cirq_ft.deprecation import deprecated_cirq_ft_class
@attr.frozen
class BiQubitsMixer(infra.GateWithRegisters):
    """Implements the COMPARE2 (Fig. 1) https://static-content.springer.com/esm/art%3A10.1038%2Fs41534-018-0071-5/MediaObjects/41534_2018_71_MOESM1_ESM.pdf

    This gates mixes the values in a way that preserves the result of comparison.
    The signature being compared are 2-qubit signature where
        x = 2*x_msb + x_lsb
        y = 2*y_msb + y_lsb
    The Gate mixes the 4 qubits so that sign(x - y) = sign(x_lsb' - y_lsb') where x_lsb' and y_lsb'
    are the final values of x_lsb' and y_lsb'.

    Note that the ancilla qubits are used to reduce the T-count and the user
    should clean the qubits at a later point in time with the adjoint gate.
    See: https://github.com/quantumlib/Cirq/pull/6313 and
    https://github.com/quantumlib/Qualtran/issues/389
    """
    adjoint: bool = False

    @cached_property
    def signature(self) -> infra.Signature:
        one_side = infra.Side.RIGHT if not self.adjoint else infra.Side.LEFT
        return infra.Signature([infra.Register('x', 2), infra.Register('y', 2), infra.Register('ancilla', 3, side=one_side)])

    def __repr__(self) -> str:
        return f'cirq_ft.algos.BiQubitsMixer({self.adjoint})'

    def decompose_from_registers(self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]) -> cirq.OP_TREE:
        x, y, ancilla = (quregs['x'], quregs['y'], quregs['ancilla'])
        x_msb, x_lsb = x
        y_msb, y_lsb = y

        def _cswap(control: cirq.Qid, a: cirq.Qid, b: cirq.Qid, aux: cirq.Qid) -> cirq.OP_TREE:
            """A CSWAP with 4T complexity and whose adjoint has 0T complexity.

                A controlled SWAP that swaps `a` and `b` based on `control`.
            It uses an extra qubit `aux` so that its adjoint would have
            a T complexity of zero.
            """
            yield cirq.CNOT(a, b)
            yield and_gate.And(adjoint=self.adjoint).on(control, b, aux)
            yield cirq.CNOT(aux, a)
            yield cirq.CNOT(a, b)

        def _decomposition():
            yield cirq.X(ancilla[0])
            yield cirq.CNOT(y_msb, x_msb)
            yield cirq.CNOT(y_lsb, x_lsb)
            yield from _cswap(x_msb, x_lsb, ancilla[0], ancilla[1])
            yield from _cswap(x_msb, y_msb, y_lsb, ancilla[2])
            yield cirq.CNOT(y_lsb, x_lsb)
        if self.adjoint:
            yield from reversed(tuple(cirq.flatten_to_ops(_decomposition())))
        else:
            yield from _decomposition()

    def __pow__(self, power: int) -> cirq.Gate:
        if power == 1:
            return self
        if power == -1:
            return BiQubitsMixer(adjoint=not self.adjoint)
        return NotImplemented

    def _t_complexity_(self) -> infra.TComplexity:
        if self.adjoint:
            return infra.TComplexity(clifford=18)
        return infra.TComplexity(t=8, clifford=28)

    def _has_unitary_(self):
        return not self.adjoint