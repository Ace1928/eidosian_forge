from typing import Optional, List, Hashable, TYPE_CHECKING
import abc
from cirq import circuits, ops, protocols, transformers
from cirq.protocols.decompose_protocol import DecomposeResult
from cirq.transformers import merge_k_qubit_gates, merge_single_qubit_gates
class TwoQubitCompilationTargetGateset(CompilationTargetGateset):
    """Abstract base class to create two-qubit target gatesets.

    This base class can be used to create two-qubit compilation target gatesets. It automatically
    implements the logic to

        1. Apply `self.preprocess_transformers` to the input circuit, which by default will:
            a) Expand composite gates acting on > 2 qubits using `cirq.expand_composite`.
            b) Merge connected components of 1 & 2 qubit unitaries into tagged
                `cirq.CircuitOperation` using `cirq.merge_k_qubit_unitaries`.

        2. Apply `self.decompose_to_target_gateset` to rewrite each operation (including merged
        connected components from 1b) using gates from this gateset.
            a) Uses `self._decompose_single_qubit_operation`, `self._decompose_two_qubit_operation`
               and `self._decompose_multi_qubit_operation` to figure out how to rewrite (merged
               connected components of) operations using only gates from this gateset.
            b) A merged connected component containing only 1 & 2q gates from this gateset is
               replaced with a more efficient rewrite using `self._decompose_two_qubit_operation`
               iff the rewritten op-tree is lesser number of 2q interactions.

            Replace connected components with inefficient implementations (higher number of 2q
               interactions) with efficient rewrites to minimize total number of 2q interactions.

        3. Apply `self.postprocess_transformers` to the transformed circuit, which by default will:
            a) Apply `cirq.merge_single_qubit_moments_to_phxz` to preserve moment structure (eg:
               alternating layers of single/two qubit gates).
            b) Apply `cirq.drop_negligible_operations` and `cirq.drop_empty_moments` to minimize
               circuit depth.

    Derived classes should simply implement `self._decompose_two_qubit_operation` abstract method
    and provide analytical decomposition of any 2q unitary using gates from the target gateset.
    """

    @property
    def num_qubits(self) -> int:
        return 2

    def decompose_to_target_gateset(self, op: 'cirq.Operation', moment_idx: int) -> DecomposeResult:
        if not 1 <= protocols.num_qubits(op) <= 2:
            return self._decompose_multi_qubit_operation(op, moment_idx)
        if protocols.num_qubits(op) == 1:
            return self._decompose_single_qubit_operation(op, moment_idx)
        new_optree = self._decompose_two_qubit_operation(op, moment_idx)
        if new_optree is NotImplemented or new_optree is None:
            return new_optree
        new_optree = [*ops.flatten_to_ops_or_moments(new_optree)]
        op_untagged = op.untagged
        old_optree = [*op_untagged.circuit] if isinstance(op_untagged, circuits.CircuitOperation) and self._intermediate_result_tag in op.tags else [op]
        old_2q_gate_count = sum((1 for o in ops.flatten_to_ops(old_optree) if len(o.qubits) == 2))
        new_2q_gate_count = sum((1 for o in ops.flatten_to_ops(new_optree) if len(o.qubits) == 2))
        switch_to_new = any((protocols.num_qubits(op) == 2 and op not in self for op in ops.flatten_to_ops(old_optree))) or new_2q_gate_count < old_2q_gate_count
        if switch_to_new:
            return new_optree
        mapped_old_optree: List['cirq.OP_TREE'] = []
        for old_op in ops.flatten_to_ops(old_optree):
            if old_op in self:
                mapped_old_optree.append(old_op)
            else:
                decomposed_op = self._decompose_single_qubit_operation(old_op, moment_idx)
                if decomposed_op is None or decomposed_op is NotImplemented:
                    return NotImplemented
                mapped_old_optree.append(decomposed_op)
        return mapped_old_optree

    def _decompose_single_qubit_operation(self, op: 'cirq.Operation', moment_idx: int) -> DecomposeResult:
        """Decomposes (connected component of) 1-qubit operations using gates from this gateset.

        By default, rewrites every operation using a single `cirq.PhasedXZGate`.

        Args:
            op: A single-qubit operation (can be a tagged `cirq.CircuitOperation` wrapping
                a connected component of single qubit unitaries).
            moment_idx: Index of the moment in which operation `op` occurs.

        Returns:
            A `cirq.OP_TREE` implementing `op` using gates from this gateset OR
            None or NotImplemented if decomposition of `op` is unknown.
        """
        return ops.PhasedXZGate.from_matrix(protocols.unitary(op)).on(op.qubits[0]) if protocols.has_unitary(op) else NotImplemented

    def _decompose_multi_qubit_operation(self, op: 'cirq.Operation', moment_idx: int) -> DecomposeResult:
        """Decomposes operations acting on more than 2 qubits using gates from this gateset.

        Args:
            op: A multi qubit (>2q) operation.
            moment_idx: Index of the moment in which operation `op` occurs.

        Returns:
            A `cirq.OP_TREE` implementing `op` using gates from this gateset OR
            None or NotImplemented if decomposition of `op` is unknown.
        """
        return NotImplemented

    @abc.abstractmethod
    def _decompose_two_qubit_operation(self, op: 'cirq.Operation', moment_idx: int) -> DecomposeResult:
        """Decomposes (connected component of) 2-qubit operations using gates from this gateset.

        Args:
            op: A two-qubit operation (can be a tagged `cirq.CircuitOperation` wrapping
                a connected component of 1 & 2  qubit unitaries).
            moment_idx: Index of the moment in which operation `op` occurs.

        Returns:
            A `cirq.OP_TREE` implementing `op` using gates from this gateset OR
            None or NotImplemented if decomposition of `op` is unknown.
        """