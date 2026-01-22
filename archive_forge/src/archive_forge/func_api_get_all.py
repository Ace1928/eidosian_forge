from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.routeros.plugins.module_utils.quoting import (
from ansible_collections.community.routeros.plugins.module_utils.api import (
import re
def api_get_all(self):
    try:
        for i in self.api_path:
            self.result['message'].append(i)
        self.return_result(False, True)
    except LibRouterosError as e:
        self.errors(e)