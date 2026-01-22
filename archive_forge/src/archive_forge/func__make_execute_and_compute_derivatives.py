from functools import wraps
from pennylane.devices import Device, DefaultExecutionConfig
from pennylane.tape import QuantumScript
def _make_execute_and_compute_derivatives(batch_execute_and_compute_derivatives):
    """Allows an ``execute_and_compute_derivatives`` method to handle individual circuits."""

    @wraps(batch_execute_and_compute_derivatives)
    def execute_and_compute_derivatives(self, circuits, execution_config=DefaultExecutionConfig):
        is_single_circuit = False
        if isinstance(circuits, QuantumScript):
            is_single_circuit = True
            circuits = (circuits,)
        results, jacs = batch_execute_and_compute_derivatives(self, circuits, execution_config)
        return (results[0], jacs[0]) if is_single_circuit else (results, jacs)
    return execute_and_compute_derivatives