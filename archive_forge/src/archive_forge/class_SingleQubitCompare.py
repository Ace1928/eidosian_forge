from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union
import attr
import cirq
from cirq._compat import cached_property
from numpy.typing import NDArray
from cirq_ft import infra
from cirq_ft.algos import and_gate
from cirq_ft.deprecation import deprecated_cirq_ft_class
@attr.frozen
class SingleQubitCompare(infra.GateWithRegisters):
    """Applies U|a>|b>|0>|0> = |a> |a=b> |(a<b)> |(a>b)>

    Source: (FIG. 3) in https://static-content.springer.com/esm/art%3A10.1038%2Fs41534-018-0071-5/MediaObjects/41534_2018_71_MOESM1_ESM.pdf
    """
    adjoint: bool = False

    @cached_property
    def signature(self) -> infra.Signature:
        one_side = infra.Side.RIGHT if not self.adjoint else infra.Side.LEFT
        return infra.Signature([infra.Register('a', 1), infra.Register('b', 1), infra.Register('less_than', 1, side=one_side), infra.Register('greater_than', 1, side=one_side)])

    def __repr__(self) -> str:
        return f'cirq_ft.algos.SingleQubitCompare({self.adjoint})'

    def decompose_from_registers(self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]) -> cirq.OP_TREE:
        a = quregs['a']
        b = quregs['b']
        less_than = quregs['less_than']
        greater_than = quregs['greater_than']

        def _decomposition() -> Iterator[cirq.Operation]:
            yield and_gate.And((0, 1), adjoint=self.adjoint).on(*a, *b, *less_than)
            yield cirq.CNOT(*less_than, *greater_than)
            yield cirq.CNOT(*b, *greater_than)
            yield cirq.CNOT(*a, *b)
            yield cirq.CNOT(*a, *greater_than)
            yield cirq.X(*b)
        if self.adjoint:
            yield from reversed(tuple(_decomposition()))
        else:
            yield from _decomposition()

    def __pow__(self, power: int) -> cirq.Gate:
        if not isinstance(power, int):
            raise ValueError('SingleQubitCompare is only defined for integer powers.')
        if power % 2 == 0:
            return cirq.IdentityGate(4)
        if power < 0:
            return SingleQubitCompare(adjoint=not self.adjoint)
        return self

    def _t_complexity_(self) -> infra.TComplexity:
        if self.adjoint:
            return infra.TComplexity(clifford=11)
        return infra.TComplexity(t=4, clifford=16)