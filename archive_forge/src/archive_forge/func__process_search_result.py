from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.errors import AnsibleActionFail, AnsibleConnectionFailure
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.validation import check_type_bool, check_type_int
from ansible.plugins.action import ActionBase
from ansible.utils.display import Display
from ..plugin_utils._reboot import reboot_host
def _process_search_result(self, result):
    self.updates = dict(((u['id'], u) for u in result['updates']))
    self.filtered_updates = dict(((u['id'], u['reasons']) for u in result['filtered']))
    for update in result['updates']:
        update_id = update['id']
        if update_id not in self.filtered_updates:
            self.selected_updates.add(update_id)