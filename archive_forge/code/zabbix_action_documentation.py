from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
from ansible.module_utils.compat.version import LooseVersion
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
Construct the user defined filter conditions to fit the Zabbix API
        requirements operations data using helper methods.

        Args:
            _formula:  zabbix condition evaluation formula
            _conditions: conditions to construct

        Returns:
            dict: user defined filter conditions
        