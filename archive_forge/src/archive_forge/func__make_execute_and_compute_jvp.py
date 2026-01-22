from functools import wraps
from pennylane.devices import Device, DefaultExecutionConfig
from pennylane.tape import QuantumScript
def _make_execute_and_compute_jvp(batch_execute_and_compute_jvp):
    """Allows an ``execute_and_compute_jvp`` method to handle individual circuits."""

    @wraps(batch_execute_and_compute_jvp)
    def execute_and_compute_jvp(self, circuits, tangents, execution_config=DefaultExecutionConfig):
        is_single_circuit = False
        if isinstance(circuits, QuantumScript):
            is_single_circuit = True
            circuits = [circuits]
            tangents = [tangents]
        results, jvps = batch_execute_and_compute_jvp(self, circuits, tangents, execution_config)
        return (results[0], jvps[0]) if is_single_circuit else (results, jvps)
    return execute_and_compute_jvp