from functools import wraps
from pennylane.devices import DefaultExecutionConfig, Device
from pennylane.tape import QuantumScript
from ..qubit.sampling import get_num_shots_and_executions
def _track_compute_vjp(untracked_compute_vjp):
    """Adds default tracking to a ``compute_vjp`` method."""

    @wraps(untracked_compute_vjp)
    def compute_vjp(self, circuits, cotangents, execution_config=DefaultExecutionConfig):
        if self.tracker.active:
            batch = (circuits,) if isinstance(circuits, QuantumScript) else circuits
            self.tracker.update(vjp_batches=1, vjps=len(batch))
            self.tracker.record()
        return untracked_compute_vjp(self, circuits, cotangents, execution_config)
    return compute_vjp