import itertools
from typing import cast, Any, Dict, List, Optional, Sequence
import cirq
from cirq.protocols.decompose_protocol import DecomposeResult
from cirq_google import ops
from cirq_google.transformers.analytical_decompositions import two_qubit_to_sycamore
class SycamoreTargetGateset(cirq.TwoQubitCompilationTargetGateset):
    """Target gateset containing Sycamore + single qubit rotations + Measurement gates."""

    def __init__(self, *, atol: float=1e-08, tabulation: Optional[cirq.TwoQubitGateTabulation]=None) -> None:
        """Inits `cirq_google.SycamoreTargetGateset`.

        Args:
            atol: A limit on the amount of absolute error introduced by the decomposition.
            tabulation: If set, a tabulation for the Sycamore gate is used for decomposing Matrix
                gates. If unset, an analytic calculation is used for Matrix gates. In both cases,
                known decompositions for gates take priority over analytical / tabulation methods.
                To get `cirq.TwoQubitGateTabulation`, call `cirq.two_qubit_gate_product_tabulation`
                with a base gate (in this case, `cirq_google.SYC`) and a maximum infidelity.
        """
        super().__init__(ops.SYC, cirq.MeasurementGate, cirq.PhasedXZGate, cirq.PhasedXPowGate, cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate, cirq.GlobalPhaseGate, name='SycamoreTargetGateset')
        self.atol = atol
        self.tabulation = tabulation

    @property
    def preprocess_transformers(self) -> List[cirq.TRANSFORMER]:
        return [cirq.create_transformer_with_kwargs(cirq.expand_composite, no_decomp=lambda op: cirq.num_qubits(op) <= self.num_qubits), cirq.create_transformer_with_kwargs(merge_swap_rzz_and_2q_unitaries, intermediate_result_tag=self._intermediate_result_tag)]

    def _decompose_two_qubit_operation(self, op: cirq.Operation, _) -> DecomposeResult:
        if not cirq.has_unitary(op):
            return NotImplemented
        known_decomp = two_qubit_to_sycamore.known_2q_op_to_sycamore_operations(op)
        if known_decomp is not None:
            return known_decomp
        if self.tabulation is not None:
            return two_qubit_to_sycamore._decompose_arbitrary_into_syc_tabulation(op, self.tabulation)
        return two_qubit_to_sycamore.two_qubit_matrix_to_sycamore_operations(op.qubits[0], op.qubits[1], cirq.unitary(op))

    def __repr__(self) -> str:
        return f'cirq_google.SycamoreTargetGateset(atol={self.atol}, tabulation={self.tabulation})'

    def _json_dict_(self) -> Dict[str, Any]:
        return {'atol': self.atol, 'tabulation': self.tabulation}

    @classmethod
    def _from_json_dict_(cls, atol, tabulation, **kwargs):
        return cls(atol=atol, tabulation=tabulation)