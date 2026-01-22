from functools import lru_cache
from typing import List, Union
import numpy as np
from qiskit.circuit import Qubit
from qiskit.circuit.operation import Operation
from qiskit.circuit.controlflow import ControlFlowOp
from qiskit.quantum_info.operators import Operator
class CommutationChecker:
    """This code is essentially copy-pasted from commutative_analysis.py.
    This code cleverly hashes commutativity and non-commutativity results between DAG nodes and seems
    quite efficient for large Clifford circuits.
    They may be other possible efficiency improvements: using rule-based commutativity analysis,
    evicting from the cache less useful entries, etc.
    """

    def __init__(self, standard_gate_commutations: dict=None, cache_max_entries: int=10 ** 6):
        super().__init__()
        if standard_gate_commutations is None:
            self._standard_commutations = {}
        else:
            self._standard_commutations = standard_gate_commutations
        self._cache_max_entries = cache_max_entries
        self._cached_commutations = {}
        self._current_cache_entries = 0
        self._cache_miss = 0
        self._cache_hit = 0

    def commute(self, op1: Operation, qargs1: List, cargs1: List, op2: Operation, qargs2: List, cargs2: List, max_num_qubits: int=3) -> bool:
        """
        Checks if two Operations commute. The return value of `True` means that the operations
        truly commute, and the return value of `False` means that either the operations do not
        commute or that the commutation check was skipped (for example, when the operations
        have conditions or have too many qubits).

        Args:
            op1: first operation.
            qargs1: first operation's qubits.
            cargs1: first operation's clbits.
            op2: second operation.
            qargs2: second operation's qubits.
            cargs2: second operation's clbits.
            max_num_qubits: the maximum number of qubits to consider, the check may be skipped if
                the number of qubits for either operation exceeds this amount.

        Returns:
            bool: whether two operations commute.
        """
        structural_commutation = _commutation_precheck(op1, qargs1, cargs1, op2, qargs2, cargs2, max_num_qubits)
        if structural_commutation is not None:
            return structural_commutation
        first_op_tuple, second_op_tuple = _order_operations(op1, qargs1, cargs1, op2, qargs2, cargs2)
        first_op, first_qargs, _ = first_op_tuple
        second_op, second_qargs, _ = second_op_tuple
        first_params = first_op.params
        second_params = second_op.params
        commutation_lookup = self.check_commutation_entries(first_op, first_qargs, second_op, second_qargs)
        if commutation_lookup is not None:
            return commutation_lookup
        is_commuting = _commute_matmul(first_op, first_qargs, second_op, second_qargs)
        if self._current_cache_entries >= self._cache_max_entries:
            self.clear_cached_commutations()
        if len(first_params) > 0 or len(second_params) > 0:
            self._cached_commutations.setdefault((first_op.name, second_op.name), {}).setdefault(_get_relative_placement(first_qargs, second_qargs), {})[_hashable_parameters(first_params), _hashable_parameters(second_params)] = is_commuting
        else:
            self._cached_commutations.setdefault((first_op.name, second_op.name), {})[_get_relative_placement(first_qargs, second_qargs)] = is_commuting
        self._current_cache_entries += 1
        return is_commuting

    def num_cached_entries(self):
        """Returns number of cached entries"""
        return self._current_cache_entries

    def clear_cached_commutations(self):
        """Clears the dictionary holding cached commutations"""
        self._current_cache_entries = 0
        self._cache_miss = 0
        self._cache_hit = 0
        self._cached_commutations = {}

    def check_commutation_entries(self, first_op: Operation, first_qargs: List, second_op: Operation, second_qargs: List) -> Union[bool, None]:
        """Returns stored commutation relation if any

        Args:
            first_op: first operation.
            first_qargs: first operation's qubits.
            second_op: second operation.
            second_qargs: second operation's qubits.

        Return:
            bool: True if the gates commute and false if it is not the case.
        """
        commutation = _query_commutation(first_op, first_qargs, second_op, second_qargs, self._standard_commutations)
        if commutation is not None:
            return commutation
        commutation = _query_commutation(first_op, first_qargs, second_op, second_qargs, self._cached_commutations)
        if commutation is None:
            self._cache_miss += 1
        else:
            self._cache_hit += 1
        return commutation