from __future__ import absolute_import, division, print_function
import base64
import traceback
from io import BytesIO
from operator import itemgetter
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
def _get_group_id(self, group_name):
    exist_group = self._zapi.hostgroup.get({'filter': {'name': group_name}})
    if exist_group:
        return exist_group[0]['groupid']
    return None