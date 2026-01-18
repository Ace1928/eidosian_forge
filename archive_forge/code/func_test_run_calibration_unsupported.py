import pytest
import numpy as np
import sympy
import cirq
from cirq_google.cloud import quantum
from cirq_google.engine.abstract_local_program import AbstractLocalProgram
from cirq_google.engine.simulated_local_job import SimulatedLocalJob
from cirq_google.engine.local_simulation_type import LocalSimulationType
def test_run_calibration_unsupported():
    program = ParentProgram([cirq.Circuit()], None)
    job = SimulatedLocalJob(job_id='test_job', processor_id='test1', parent_program=program, repetitions=100, sweeps=[{}])
    with pytest.raises(NotImplementedError):
        _ = job.calibration_results()