from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
from ansible_collections.netapp.ontap.plugins.module_utils import rest_vserver
def apply_policy(self):
    """
        Apply a Scanner policy to a Scanner pool
        :return: nothing
        """
    apply_policy_obj = netapp_utils.zapi.NaElement('vscan-scanner-pool-apply-policy')
    apply_policy_obj.add_new_child('scanner-policy', self.parameters['scanner_policy'])
    apply_policy_obj.add_new_child('scanner-pool', self.parameters['scanner_pool'])
    try:
        self.server.invoke_successfully(apply_policy_obj, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error appling policy %s to pool %s: %s' % (self.parameters['scanner_policy'], self.parameters['scanner_policy'], to_native(error)), exception=traceback.format_exc())