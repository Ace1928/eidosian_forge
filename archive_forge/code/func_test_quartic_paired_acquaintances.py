import itertools
import random
import pytest
import cirq
import cirq.contrib.acquaintance as cca
@pytest.mark.parametrize('index_pairs', [random_index_pairs(n_pairs) for n_pairs in range(2, 7) for _ in range(2)])
def test_quartic_paired_acquaintances(index_pairs):
    n_pairs = len(index_pairs)
    qubit_pairs = tuple((tuple((cirq.LineQubit(x) for x in index_pair)) for index_pair in index_pairs))
    strategy, qubits = cca.quartic_paired_acquaintance_strategy(qubit_pairs)
    initial_mapping = {q: q.x for q in qubits}
    opps = cca.get_logical_acquaintance_opportunities(strategy, initial_mapping)
    assert set((len(opp) for opp in opps)) == set([2, 4])
    quadratic_opps = set((opp for opp in opps if len(opp) == 2))
    expected_quadratic_opps = set((frozenset(index_pair) for index_pair in itertools.combinations(range(2 * n_pairs), 2)))
    assert quadratic_opps == expected_quadratic_opps
    quartic_opps = set((opp for opp in opps if len(opp) == 4))
    expected_quartic_opps = set((frozenset(I + J) for I, J in itertools.combinations(index_pairs, 2)))
    assert quartic_opps == expected_quartic_opps