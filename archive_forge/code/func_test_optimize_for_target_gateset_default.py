import cirq
from cirq.protocols.decompose_protocol import DecomposeResult
from cirq.transformers.optimize_for_target_gateset import _decompose_operations_to_target_gateset
import pytest
def test_optimize_for_target_gateset_default():
    q = cirq.LineQubit.range(2)
    c_orig = cirq.Circuit(cirq.T(q[0]), cirq.SWAP(*q), cirq.T(q[0]), cirq.SWAP(*q).with_tags('ignore'))
    context = cirq.TransformerContext(tags_to_ignore=('ignore',))
    c_new = cirq.optimize_for_target_gateset(c_orig, context=context)
    cirq.testing.assert_has_diagram(c_new, "\n0: ───T────────────@───Y^-0.5───@───Y^0.5────@───────────T───×['ignore']───\n                   │            │            │               │\n1: ───────Y^-0.5───@───Y^0.5────@───Y^-0.5───@───Y^0.5───────×─────────────\n")
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(c_orig, c_new, atol=1e-06)