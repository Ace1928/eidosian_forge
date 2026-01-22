from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _get_dcs_id(self):
    return get_id_by_name(self._get_dcs_service(), self.param('data_center'))