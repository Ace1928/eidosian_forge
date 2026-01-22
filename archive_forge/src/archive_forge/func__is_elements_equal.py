from __future__ import absolute_import, division, print_function
import base64
import traceback
from io import BytesIO
from operator import itemgetter
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def _is_elements_equal(self, generated_elements, exist_elements):
    if len(generated_elements) != len(exist_elements):
        return False
    generated_elements_sorted = sorted(generated_elements, key=lambda k: k.values()[0])
    exist_elements_sorted = sorted(exist_elements, key=lambda k: k.values()[0])
    for generated_element, exist_element in zip(generated_elements_sorted, exist_elements_sorted):
        if not self._is_dicts_equal(generated_element, exist_element, ['selementid']):
            return False
    return True