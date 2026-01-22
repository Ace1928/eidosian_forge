from functools import wraps
from pennylane.devices import Device, DefaultExecutionConfig
from pennylane.tape import QuantumScript
def _make_execute(batch_execute):
    """Allows an ``execute`` function to handle individual circuits."""

    @wraps(batch_execute)
    def execute(self, circuits, execution_config=DefaultExecutionConfig):
        is_single_circuit = False
        if isinstance(circuits, QuantumScript):
            is_single_circuit = True
            circuits = (circuits,)
        results = batch_execute(self, circuits, execution_config)
        return results[0] if is_single_circuit else results
    return execute