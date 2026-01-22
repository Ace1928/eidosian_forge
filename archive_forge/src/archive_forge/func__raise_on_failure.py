import datetime
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import duet
from google.protobuf import any_pb2
import cirq
from cirq_google.engine import abstract_job, calibration, engine_client
from cirq_google.engine.calibration_result import CalibrationResult
from cirq_google.cloud import quantum
from cirq_google.engine.result_type import ResultType
from cirq_google.engine.engine_result import EngineResult
from cirq_google.api import v1, v2
def _raise_on_failure(job: quantum.QuantumJob) -> None:
    execution_status = job.execution_status
    state = execution_status.state
    name = job.name
    if state != quantum.ExecutionStatus.State.SUCCESS:
        if state == quantum.ExecutionStatus.State.FAILURE:
            processor = execution_status.processor_name or 'UNKNOWN'
            error_code = execution_status.failure.error_code
            error_message = execution_status.failure.error_message
            raise RuntimeError(f'Job {name} on processor {processor} failed. {error_code.name}: {error_message}')
        elif state in TERMINAL_STATES:
            raise RuntimeError(f'Job {name} failed in state {state.name}.')
        else:
            raise RuntimeError(f'Timed out waiting for results. Job {name} is in state {state.name}')