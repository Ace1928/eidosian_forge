from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
Ensures interfaces object is properly formatted before submitting it to API.

        Args:
            interfaces (list): list of dictionaries for each interface present on the host.

        Returns:
            (interfaces, ip) - where interfaces is original list reformated into a valid format
                and ip is any IP address found on interface of type agent (printing purposes only).
        