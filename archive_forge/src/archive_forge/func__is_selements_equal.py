from __future__ import absolute_import, division, print_function
import base64
import traceback
from io import BytesIO
from operator import itemgetter
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def _is_selements_equal(self, generated_selements, exist_selements):
    if len(generated_selements) != len(exist_selements):
        return False
    generated_selements_sorted = sorted(generated_selements, key=itemgetter(*self.selements_sort_keys))
    exist_selements_sorted = sorted(exist_selements, key=itemgetter(*self.selements_sort_keys))
    for generated_selement, exist_selement in zip(generated_selements_sorted, exist_selements_sorted):
        if not self._is_elements_equal(generated_selement.get('elements', []), exist_selement.get('elements', [])):
            return False
        if not self._is_dicts_equal(generated_selement, exist_selement, ['selementid']):
            return False
        if not self._is_urls_equal(generated_selement.get('urls', []), exist_selement.get('urls', [])):
            return False
    return True