from __future__ import absolute_import, division, print_function
import time
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail, AnsibleConnectionFailure
from ansible.utils.vars import merge_hash
from ansible.utils.display import Display
def _async_result(self, async_status_args, task_vars, timeout):
    """
        Retrieve results of the asynchronous task, and display them in place of
        the async wrapper results (those with the ansible_job_id key).
        """
    async_status = self._task.copy()
    async_status.args = async_status_args
    async_status.action = 'ansible.builtin.async_status'
    async_status.async_val = 0
    async_action = self._shared_loader_obj.action_loader.get(async_status.action, task=async_status, connection=self._connection, play_context=self._play_context, loader=self._loader, templar=self._templar, shared_loader_obj=self._shared_loader_obj)
    if async_status.args['mode'] == 'cleanup':
        return async_action.run(task_vars=task_vars)
    for dummy in range(max(1, timeout)):
        async_result = async_action.run(task_vars=task_vars)
        if async_result.get('finished', 0) == 1:
            break
        time.sleep(min(1, timeout))
    return async_result