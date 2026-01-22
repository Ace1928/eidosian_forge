import contextlib
import threading
class PreemptionSaveContext(threading.local):
    """A context for saving checkpoint upon preemption."""

    def __init__(self):
        super().__init__()
        self._in_preemption_save_context = False

    def enter_preemption_save_context(self):
        self._in_preemption_save_context = True

    def exit_preemption_save_context(self):
        self._in_preemption_save_context = False

    def in_preemption_save_context(self):
        return self._in_preemption_save_context