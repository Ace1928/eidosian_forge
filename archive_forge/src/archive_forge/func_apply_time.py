from __future__ import absolute_import, division, print_function
import json
import time
from urllib.error import HTTPError, URLError
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.compat.version import LooseVersion
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible_collections.dellemc.openmanage.plugins.module_utils.idrac_redfish import (
from ansible_collections.dellemc.openmanage.plugins.module_utils.utils import (
def apply_time(self, setting_uri):
    resp = get_dynamic_uri(self.idrac, setting_uri, '@Redfish.Settings')
    rf_settings = resp.get('SupportedApplyTimes', [])
    apply_time = self.module.params.get('apply_time', {})
    rf_set = self.__get_redfish_apply_time(apply_time, rf_settings)
    return rf_set