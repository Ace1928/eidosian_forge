import itertools
import random
import pytest
import cirq
import cirq.contrib.acquaintance as cca
@pytest.mark.parametrize('left_part_lens,right_part_lens', [tuple((random_part_lens(7, 2) for _ in ('left', 'right'))) for _ in range(5)])
def test_shift_swap_network_gate_acquaintance_opps(left_part_lens, right_part_lens):
    gate = cca.ShiftSwapNetworkGate(left_part_lens, right_part_lens)
    n_qubits = gate.qubit_count()
    qubits = cirq.LineQubit.range(n_qubits)
    strategy = cirq.Circuit(gate(*qubits))
    initial_mapping = {q: i for i, q in enumerate(qubits)}
    actual_opps = cca.get_logical_acquaintance_opportunities(strategy, initial_mapping)
    i = 0
    sides = ('left', 'right')
    parts = {side: [] for side in sides}
    for side, part_lens in zip(sides, (left_part_lens, right_part_lens)):
        for part_len in part_lens:
            parts[side].append(set(range(i, i + part_len)))
            i += part_len
    expected_opps = set((frozenset(left_part | right_part) for left_part, right_part in itertools.product(parts['left'], parts['right'])))
    assert actual_opps == expected_opps