from __future__ import absolute_import, division, print_function
import base64
import traceback
from io import BytesIO
from operator import itemgetter
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def _is_urls_equal(self, generated_urls, exist_urls):
    if len(generated_urls) != len(exist_urls):
        return False
    generated_urls_sorted = sorted(generated_urls, key=itemgetter('name', 'url'))
    exist_urls_sorted = sorted(exist_urls, key=itemgetter('name', 'url'))
    for generated_url, exist_url in zip(generated_urls_sorted, exist_urls_sorted):
        if not self._is_dicts_equal(generated_url, exist_url, ['selementid']):
            return False
    return True