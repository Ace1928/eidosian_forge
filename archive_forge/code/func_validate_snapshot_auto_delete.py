from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.rest_application import RestApplication
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def validate_snapshot_auto_delete(self):
    if 'snapshot_auto_delete' in self.parameters:
        for key in self.parameters['snapshot_auto_delete']:
            if key not in ['commitment', 'trigger', 'target_free_space', 'delete_order', 'defer_delete', 'prefix', 'destroy_list', 'state']:
                self.module.fail_json(msg="snapshot_auto_delete option '%s' is not valid." % key)