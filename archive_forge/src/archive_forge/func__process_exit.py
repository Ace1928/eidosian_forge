from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.errors import AnsibleActionFail, AnsibleConnectionFailure
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.validation import check_type_bool, check_type_int
from ansible.plugins.action import ActionBase
from ansible.utils.display import Display
from ..plugin_utils._reboot import reboot_host
def _process_exit(self, result, inventory_hostname):
    display.vv('Received final progress result from update task', host=inventory_hostname)
    self.changed = result['changed']
    self.reboot_required = result['reboot_required']
    self.failed = result['failed']
    if result.get('exception', None):
        self.msg = result['exception']['message']
        self.exception = result['exception'].get('exception', None)
        if 'hresult' in result['exception']:
            self.hresult = result['exception']['hresult'] & 4294967295