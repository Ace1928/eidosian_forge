from functools import wraps
from pennylane.devices import DefaultExecutionConfig, Device
from pennylane.tape import QuantumScript
from ..qubit.sampling import get_num_shots_and_executions
def _track_compute_jvp(untracked_compute_jvp):
    """Adds default tracking to a ``compute_jvp`` method."""

    @wraps(untracked_compute_jvp)
    def compute_jvp(self, circuits, tangents, execution_config=DefaultExecutionConfig):
        if self.tracker.active:
            batch = (circuits,) if isinstance(circuits, QuantumScript) else circuits
            self.tracker.update(jvp_batches=1, jvps=len(batch))
            self.tracker.record()
        return untracked_compute_jvp(self, circuits, tangents, execution_config)
    return compute_jvp