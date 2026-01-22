import concurrent.futures
import datetime
from typing import cast, List, Optional, Sequence, Tuple
import duet
import cirq
from cirq_google.cloud import quantum
from cirq_google.engine.calibration_result import CalibrationResult
from cirq_google.engine.abstract_local_job import AbstractLocalJob
from cirq_google.engine.local_simulation_type import LocalSimulationType
from cirq_google.engine.engine_result import EngineResult
def _execute_results(self) -> Sequence[Sequence[EngineResult]]:
    """Executes the circuit and sweeps on the sampler.

        For synchronous execution, this is called when the results()
        function is called.  For asynchronous execution, this function
        is run in a thread pool that begins when the object is
        instantiated.

        Returns: a List of results from the sweep's execution.
        """
    reps, sweeps = self.get_repetitions_and_sweeps()
    parent = self.program()
    batch_size = parent.batch_size()
    try:
        self._state = quantum.ExecutionStatus.State.RUNNING
        programs = [parent.get_circuit(n) for n in range(batch_size)]
        batch_results = self._sampler.run_batch(programs=programs, params_list=cast(List[cirq.Sweepable], sweeps), repetitions=reps)
        batch_engine_results = _to_engine_results(batch_results, job_id=self.id())
        self._state = quantum.ExecutionStatus.State.SUCCESS
        return batch_engine_results
    except Exception as e:
        self._failure_code = '500'
        self._failure_message = str(e)
        self._state = quantum.ExecutionStatus.State.FAILURE
        raise e