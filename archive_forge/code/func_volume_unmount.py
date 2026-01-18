from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_volume
def volume_unmount(self, current):
    """
        Unmount FlexCache volume at destination cluster
        """
    if self.use_rest:
        self.rest_unmount_volume(current)
    else:
        options = {'volume-name': self.parameters['name']}
        xml = netapp_utils.zapi.NaElement.create_node_with_children('volume-unmount', **options)
        try:
            self.server.invoke_successfully(xml, enable_tunneling=True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error unmounting FlexCache volume: %s' % to_native(error), exception=traceback.format_exc())