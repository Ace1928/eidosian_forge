import glob
import itertools
from typing import Iterable
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import cirq
import cirq.experiments.random_quantum_circuit_generation as rqcg
from cirq.experiments.xeb_sampling import sample_2q_xeb_circuits
def test_sample_2q_parallel_xeb_circuits(tmpdir):
    circuits = rqcg.generate_library_of_2q_circuits(n_library_circuits=5, two_qubit_gate=cirq.ISWAP ** 0.5, max_cycle_depth=10)
    cycle_depths = [5, 10]
    graph = _gridqubits_to_graph_device(cirq.GridQubit.rect(3, 2))
    combs = rqcg.get_random_combinations_for_device(n_library_circuits=len(circuits), n_combinations=5, device_graph=graph, random_state=10)
    df = sample_2q_xeb_circuits(sampler=cirq.Simulator(), circuits=circuits, cycle_depths=cycle_depths, combinations_by_layer=combs, dataset_directory=f'{tmpdir}/my_dataset')
    n_pairs = sum((len(c.pairs) for c in combs))
    assert len(df) == len(cycle_depths) * len(circuits) * n_pairs
    for (circuit_i, cycle_depth), row in df.iterrows():
        assert 0 <= circuit_i < len(circuits)
        assert cycle_depth in cycle_depths
        assert len(row['sampled_probs']) == 4
        assert np.isclose(np.sum(row['sampled_probs']), 1)
        assert 0 <= row['layer_i'] < 4
        assert 0 <= row['pair_i'] < 2
    assert len(df['pair'].unique()) == 7
    chunks = [record for fn in glob.glob(f'{tmpdir}/my_dataset/*') for record in cirq.read_json(fn)]
    df2 = pd.DataFrame(chunks).set_index(['circuit_i', 'cycle_depth'])
    df2['pair'] = [tuple(row['pair']) for _, row in df2.iterrows()]
    actual_index_names = ['layer_i', 'pair_i', 'combination_i', 'cycle_depth']
    _assert_frame_approx_equal(df.reset_index().set_index(actual_index_names), df2.reset_index().set_index(actual_index_names), atol=1e-05)