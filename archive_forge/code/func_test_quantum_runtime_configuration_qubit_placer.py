import datetime
import glob
import re
import time
import uuid
from typing import List, cast, Any
import numpy as np
import pytest
import cirq
import cirq_google as cg
from cirq_google.workflow.quantum_executable_test import _get_quantum_executables, _get_example_spec
from cirq_google.workflow.quantum_runtime import _time_into_runtime_info
def test_quantum_runtime_configuration_qubit_placer(rt_config):
    device = rt_config.processor_record.get_device()
    c, _ = rt_config.qubit_placer.place_circuit(cirq.Circuit(cirq.measure(cirq.LineQubit(0), cirq.LineQubit(1), key='z')), problem_topology=cirq.LineTopology(n_nodes=2), shared_rt_info=cg.SharedRuntimeInfo(run_id=rt_config.run_id, device=device), rs=np.random.RandomState(rt_config.random_seed))
    if isinstance(rt_config.qubit_placer, cg.NaiveQubitPlacer):
        assert all((isinstance(q, cirq.LineQubit) for q in c.all_qubits()))
    else:
        assert all((isinstance(q, cirq.GridQubit) for q in c.all_qubits()))