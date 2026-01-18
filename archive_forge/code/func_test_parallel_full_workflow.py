import itertools
import multiprocessing
from typing import Iterable
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import cirq
import cirq.experiments.random_quantum_circuit_generation as rqcg
from cirq.experiments.xeb_fitting import (
from cirq.experiments.xeb_sampling import sample_2q_xeb_circuits
@pytest.mark.parametrize('use_pool', (True, False))
def test_parallel_full_workflow(use_pool):
    circuits = rqcg.generate_library_of_2q_circuits(n_library_circuits=5, two_qubit_gate=cirq.ISWAP ** 0.5, max_cycle_depth=4, random_state=8675309)
    cycle_depths = [2, 3, 4]
    graph = _gridqubits_to_graph_device(cirq.GridQubit.rect(2, 2))
    combs = rqcg.get_random_combinations_for_device(n_library_circuits=len(circuits), n_combinations=2, device_graph=graph, random_state=10)
    sampled_df = sample_2q_xeb_circuits(sampler=cirq.Simulator(), circuits=circuits, cycle_depths=cycle_depths, combinations_by_layer=combs)
    if use_pool:
        pool = multiprocessing.Pool()
    else:
        pool = None
    fids_df_0 = benchmark_2q_xeb_fidelities(sampled_df=sampled_df, circuits=circuits, cycle_depths=cycle_depths, pool=pool)
    options = SqrtISwapXEBOptions(characterize_zeta=False, characterize_gamma=False, characterize_chi=False)
    p_circuits = [parameterize_circuit(circuit, options) for circuit in circuits]
    result = characterize_phased_fsim_parameters_with_xeb_by_pair(sampled_df=sampled_df, parameterized_circuits=p_circuits, cycle_depths=cycle_depths, options=options, fatol=0.05, xatol=0.05, pool=pool)
    if pool is not None:
        pool.terminate()
    assert len(result.optimization_results) == graph.number_of_edges()
    for opt_res in result.optimization_results.values():
        assert np.abs(opt_res.fun) < 0.1
    assert len(result.fidelities_df) == len(cycle_depths) * graph.number_of_edges()
    assert np.all(result.fidelities_df['fidelity'] > 0.9)
    before_after_df = before_and_after_characterization(fids_df_0, characterization_result=result)
    for _, row in before_after_df.iterrows():
        assert len(row['fidelities_0']) == len(cycle_depths)
        assert len(row['fidelities_c']) == len(cycle_depths)
        assert 0 <= row['a_0'] <= 1
        assert 0 <= row['a_c'] <= 1
        assert 0 <= row['layer_fid_0'] <= 1
        assert 0 <= row['layer_fid_c'] <= 1