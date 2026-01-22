from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def acp_firmware_upgrade(self):
    """
        Upgrade shelf firmware image
        """
    acp_firmware_update_info = netapp_utils.zapi.NaElement('storage-shelf-acp-firmware-update')
    acp_firmware_update_info.add_new_child('node-name', self.parameters['node'])
    try:
        self.server.invoke_successfully(acp_firmware_update_info, enable_tunneling=True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error updating acp firmware image : %s' % to_native(error), exception=traceback.format_exc())