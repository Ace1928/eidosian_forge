from __future__ import absolute_import, division, print_function
import base64
import traceback
from io import BytesIO
from operator import itemgetter
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def _is_triggers_equal(self, generated_triggers, exist_triggers):
    if len(generated_triggers) != len(exist_triggers):
        return False
    generated_triggers_sorted = sorted(generated_triggers, key=itemgetter('triggerid'))
    exist_triggers_sorted = sorted(exist_triggers, key=itemgetter('triggerid'))
    for generated_trigger, exist_trigger in zip(generated_triggers_sorted, exist_triggers_sorted):
        if not self._is_dicts_equal(generated_trigger, exist_trigger):
            return False
    return True