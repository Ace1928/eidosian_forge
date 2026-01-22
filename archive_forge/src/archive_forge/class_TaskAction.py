import functools
from taskflow.engines.action_engine.actions import base
from taskflow import logging
from taskflow import states
from taskflow import task as task_atom
from taskflow.types import failure
class TaskAction(base.Action):
    """An action that handles scheduling, state changes, ... of task atoms."""

    def __init__(self, storage, notifier, task_executor):
        super(TaskAction, self).__init__(storage, notifier)
        self._task_executor = task_executor

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

    def change_state(self, task, state, progress=None, result=base.Action.NO_RESULT):
        old_state = self._storage.get_atom_state(task.name)
        if self._is_identity_transition(old_state, state, task, progress=progress):
            return
        if state in self.SAVE_RESULT_STATES:
            save_result = None
            if result is not self.NO_RESULT:
                save_result = result
            self._storage.save(task.name, save_result, state)
        else:
            self._storage.set_atom_state(task.name, state)
        if progress is not None:
            self._storage.set_task_progress(task.name, progress)
        task_uuid = self._storage.get_atom_uuid(task.name)
        details = {'task_name': task.name, 'task_uuid': task_uuid, 'old_state': old_state}
        if result is not self.NO_RESULT:
            details['result'] = result
        self._notifier.notify(state, details)
        if progress is not None:
            task.update_progress(progress)

    def _on_update_progress(self, task, event_type, details):
        """Should be called when task updates its progress."""
        try:
            progress = details.pop('progress')
        except KeyError:
            pass
        else:
            try:
                self._storage.set_task_progress(task.name, progress, details=details)
            except Exception:
                LOG.exception('Failed setting task progress for %s to %0.3f', task, progress)

    def schedule_execution(self, task):
        self.change_state(task, states.RUNNING, progress=0.0)
        arguments = self._storage.fetch_mapped_args(task.rebind, atom_name=task.name, optional_args=task.optional)
        if task.notifier.can_be_registered(task_atom.EVENT_UPDATE_PROGRESS):
            progress_callback = functools.partial(self._on_update_progress, task)
        else:
            progress_callback = None
        task_uuid = self._storage.get_atom_uuid(task.name)
        return self._task_executor.execute_task(task, task_uuid, arguments, progress_callback=progress_callback)

    def complete_execution(self, task, result):
        if isinstance(result, failure.Failure):
            self.change_state(task, states.FAILURE, result=result)
        else:
            self.change_state(task, states.SUCCESS, result=result, progress=1.0)

    def schedule_reversion(self, task):
        self.change_state(task, states.REVERTING, progress=0.0)
        arguments = self._storage.fetch_mapped_args(task.revert_rebind, atom_name=task.name, optional_args=task.revert_optional)
        task_uuid = self._storage.get_atom_uuid(task.name)
        task_result = self._storage.get(task.name)
        failures = self._storage.get_failures()
        if task.notifier.can_be_registered(task_atom.EVENT_UPDATE_PROGRESS):
            progress_callback = functools.partial(self._on_update_progress, task)
        else:
            progress_callback = None
        return self._task_executor.revert_task(task, task_uuid, arguments, task_result, failures, progress_callback=progress_callback)

    def complete_reversion(self, task, result):
        if isinstance(result, failure.Failure):
            self.change_state(task, states.REVERT_FAILURE, result=result)
        else:
            self.change_state(task, states.REVERTED, progress=1.0, result=result)