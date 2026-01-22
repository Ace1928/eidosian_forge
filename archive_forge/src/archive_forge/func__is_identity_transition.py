import functools
from taskflow.engines.action_engine.actions import base
from taskflow import logging
from taskflow import states
from taskflow import task as task_atom
from taskflow.types import failure
def _is_identity_transition(self, old_state, state, task, progress=None):
    if state in self.SAVE_RESULT_STATES:
        return False
    if state != old_state:
        return False
    if progress is None:
        return False
    old_progress = self._storage.get_task_progress(task.name)
    if old_progress != progress:
        return False
    return True