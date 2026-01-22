import itertools
from typing import Any, Dict, Optional, Sequence, Type, Union
import numpy as np
import sympy
from cirq import ops, protocols, value
from cirq.testing.consistent_act_on import assert_all_implemented_act_on_effects_match_unitary
from cirq.testing.circuit_compare import (
from cirq.testing.consistent_decomposition import (
from cirq.testing.consistent_phase_by import assert_phase_by_is_consistent_with_unitary
from cirq.testing.consistent_qasm import assert_qasm_is_consistent_with_unitary
from cirq.testing.consistent_pauli_expansion import (
from cirq.testing.consistent_resolve_parameters import assert_consistent_resolve_parameters
from cirq.testing.consistent_specified_has_unitary import assert_specifies_has_unitary_if_unitary
from cirq.testing.equivalent_repr_eval import assert_equivalent_repr
from cirq.testing.consistent_controlled_gate_op import (
from cirq.testing.consistent_unitary import assert_unitary_is_consistent
def _assert_meets_standards_helper(val: Any, *, ignoring_global_phase: bool, setup_code: str, global_vals: Optional[Dict[str, Any]], local_vals: Optional[Dict[str, Any]], ignore_decompose_to_default_gateset: bool) -> None:
    __tracebackhide__ = True
    assert_consistent_resolve_parameters(val)
    assert_specifies_has_unitary_if_unitary(val)
    assert_has_consistent_qid_shape(val)
    assert_has_consistent_apply_unitary(val)
    assert_all_implemented_act_on_effects_match_unitary(val)
    assert_qasm_is_consistent_with_unitary(val)
    assert_has_consistent_trace_distance_bound(val)
    assert_decompose_is_consistent_with_unitary(val, ignoring_global_phase=ignoring_global_phase)
    assert_unitary_is_consistent(val, ignoring_global_phase=ignoring_global_phase)
    if not ignore_decompose_to_default_gateset:
        assert_decompose_ends_at_default_gateset(val)
    assert_phase_by_is_consistent_with_unitary(val)
    assert_pauli_expansion_is_consistent_with_unitary(val)
    assert_equivalent_repr(val, setup_code=setup_code, global_vals=global_vals, local_vals=local_vals)
    assert protocols.measurement_key_objs(val) == protocols.measurement_key_names(val)
    if isinstance(val, ops.EigenGate):
        assert_eigen_shifts_is_consistent_with_eigen_components(val)
    if isinstance(val, ops.Gate) and protocols.has_mixture(val):
        assert_controlled_and_controlled_by_identical(val)
        if protocols.has_unitary(val):
            assert_controlled_unitary_consistent(val)