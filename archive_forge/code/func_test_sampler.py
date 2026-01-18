import datetime
from typing import Dict, Optional, Union
import pytest
import cirq
import cirq_google
import sympy
import numpy as np
from cirq_google.api import v2
from cirq_google.engine.abstract_local_job_test import NothingJob
from cirq_google.engine.abstract_local_program_test import NothingProgram
from cirq_google.engine.abstract_local_processor import AbstractLocalProcessor
from cirq_google.engine.abstract_program import AbstractProgram
from cirq_google.engine.simulated_local_engine import SimulatedLocalEngine
from cirq_google.engine.simulated_local_processor import SimulatedLocalProcessor
def test_sampler():
    engine = SimulatedLocalEngine([SimulatedLocalProcessor(processor_id='tester')])
    q = cirq.GridQubit(5, 4)
    circuit = cirq.Circuit(cirq.X(q) ** sympy.Symbol('t'), cirq.measure(q, key='m'))
    sweep = cirq.Points(key='t', points=[1, 0])
    results = engine.get_sampler('tester').run_sweep(circuit, params=sweep, repetitions=100)
    assert np.all(results[0].measurements['m'] == 1)
    assert np.all(results[1].measurements['m'] == 0)