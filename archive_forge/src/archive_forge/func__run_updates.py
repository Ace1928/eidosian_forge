from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.errors import AnsibleActionFail, AnsibleConnectionFailure
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.validation import check_type_bool, check_type_int
from ansible.plugins.action import ActionBase
from ansible.utils.display import Display
from ..plugin_utils._reboot import reboot_host
def _run_updates(self, task_vars, module_options):
    """Runs the win_updates module and returns the raw results from that task."""
    inventory_hostname = task_vars.get('inventory_hostname', None)
    display.vv('Starting update task', host=inventory_hostname)
    start_result = self._execute_win_updates(task_vars=task_vars, module_options=module_options)
    cancel_options = start_result.pop('cancel_options', {})
    poll_options = start_result.pop('poll_options', {})
    display.vv('Starting polling for update results', host=inventory_hostname)
    update_result = UpdateResult()
    has_errored = False
    try:
        polling = True
        while polling:
            poll_result = self._execute_win_updates(task_vars, operation='poll', operation_options=poll_options, retry_on_failure=True)
            for progress in poll_result['output']:
                task = progress['task']
                update_result.process_result(task, progress['result'], inventory_hostname)
                if task == 'exit':
                    polling = False
    except Exception as e:
        display.warning('Unknown failure when polling update result - attempting to cancel task: %s' % to_text(e))
        has_errored = True
        raise
    finally:
        try:
            self._execute_win_updates(task_vars, operation='cancel', operation_options=cancel_options)
        except Exception as e:
            if has_errored:
                display.warning('Unknown failure when cancelling update task: %s' % to_text(e))
            else:
                raise
    return update_result