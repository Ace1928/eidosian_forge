from typing import Sequence, Union, Tuple
from numpy.typing import NDArray
import attr
import cirq
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import multi_control_multi_target_pauli as mcmtp
@attr.frozen
class MultiTargetCSwap(infra.GateWithRegisters):
    """Implements a multi-target controlled swap unitary $CSWAP_n = |0><0| I + |1><1| SWAP_n$.

    This decomposes into a qubitwise SWAP on the two target signature, and takes 14*n T-gates.

    References:
        [Trading T-gates for dirty qubits in state preparation and unitary synthesis]
        (https://arxiv.org/abs/1812.00954).
        Low et. al. 2018. See Appendix B.2.c.
    """
    bitsize: int

    @classmethod
    def make_on(cls, **quregs: Union[Sequence[cirq.Qid], NDArray[cirq.Qid]]) -> cirq.Operation:
        """Helper constructor to automatically deduce bitsize attributes."""
        return cls(bitsize=len(quregs['target_x'])).on_registers(**quregs)

    @cached_property
    def signature(self) -> infra.Signature:
        return infra.Signature.build(control=1, target_x=self.bitsize, target_y=self.bitsize)

    def decompose_from_registers(self, *, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]) -> cirq.OP_TREE:
        control, target_x, target_y = (quregs['control'], quregs['target_x'], quregs['target_y'])
        yield [cirq.CSWAP(*control, t_x, t_y) for t_x, t_y in zip(target_x, target_y)]

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        if not args.use_unicode_characters:
            return cirq.CircuitDiagramInfo(('@',) + ('swap_x',) * self.bitsize + ('swap_y',) * self.bitsize)
        return cirq.CircuitDiagramInfo(('@',) + ('×(x)',) * self.bitsize + ('×(y)',) * self.bitsize)

    def __repr__(self) -> str:
        return f'cirq_ft.MultiTargetCSwap({self.bitsize})'

    def _t_complexity_(self) -> infra.TComplexity:
        return infra.TComplexity(t=7 * self.bitsize, clifford=10 * self.bitsize)