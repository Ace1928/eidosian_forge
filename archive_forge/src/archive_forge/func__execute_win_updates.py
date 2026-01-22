from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.errors import AnsibleActionFail, AnsibleConnectionFailure
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.validation import check_type_bool, check_type_int
from ansible.plugins.action import ActionBase
from ansible.utils.display import Display
from ..plugin_utils._reboot import reboot_host
def _execute_win_updates(self, task_vars, module_options=None, operation='start', operation_options=None, retry_on_failure=False):
    final_options = (module_options or {}).copy()
    final_options['_operation'] = operation
    final_options['_operation_options'] = operation_options or {}
    result = {}
    for idx in range(2):
        try:
            result = self._execute_module(module_name='ansible.windows.win_updates', module_args=final_options, task_vars=task_vars)
            break
        except (AnsibleConnectionFailure, RequestTimeoutException) as e:
            if not retry_on_failure or idx == 1:
                raise
            display.warning('Connection failure when polling update result - attempting to retry: %s' % to_text(e))
    if 'invocation' in result and (not self._invocation):
        self._invocation = result['invocation']
    if result.get('failed', False):
        msg = result.get('msg', f'Failure while running win_updates {operation}')
        extra_result = {}
        if 'rc' in result:
            extra_result['rc'] = result['rc']
        if 'stdout' in result:
            extra_result['stdout'] = result['stdout']
        if 'stderr' in result:
            extra_result['stderr'] = result['stderr']
        raise _ReturnResultException(msg, exception=result.get('exception', None), **extra_result)
    for w in result.get('warnings', []):
        display.warning(w)
    return result