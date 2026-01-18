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
def test_sample_2q_xeb_circuits_no_progress(capsys):
    qubits = cirq.LineQubit.range(2)
    circuits = [cirq.testing.random_circuit(qubits, n_moments=7, op_density=0.8, random_state=52)]
    cycle_depths = np.arange(3, 4)
    _ = sample_2q_xeb_circuits(sampler=cirq.Simulator(), circuits=circuits, cycle_depths=cycle_depths, progress_bar=None)
    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err == ''