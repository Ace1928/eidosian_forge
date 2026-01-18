import pytest
import numpy as np
import sympy
import cirq
from cirq_google.cloud import quantum
from cirq_google.engine.abstract_local_program import AbstractLocalProgram
from cirq_google.engine.simulated_local_job import SimulatedLocalJob
from cirq_google.engine.local_simulation_type import LocalSimulationType
def test_run_sweep():
    program = ParentProgram([cirq.Circuit(cirq.X(Q) ** sympy.Symbol('t'), cirq.measure(Q, key='m'))], None)
    job = SimulatedLocalJob(job_id='test_job', processor_id='test1', parent_program=program, repetitions=100, sweeps=[cirq.Points(key='t', points=[1, 0])])
    assert job.execution_status() == quantum.ExecutionStatus.State.READY
    results = job.results()
    assert np.all(results[0].measurements['m'] == 1)
    assert np.all(results[1].measurements['m'] == 0)
    assert job.execution_status() == quantum.ExecutionStatus.State.SUCCESS