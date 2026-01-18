from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.elementsw.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_elementsw_module import NaElementSWModule
from ansible_collections.netapp.elementsw.plugins.module_utils.netapp_module import NetAppModule
def unpair_volumes(self):
    """
            Delete volume pair
        """
    try:
        self.elem.remove_volume_pair(volume_id=self.parameters['src_vol_id'])
        self.dest_elem.remove_volume_pair(volume_id=self.parameters['dest_vol_id'])
    except solidfire.common.ApiServerError as err:
        self.module.fail_json(msg='Error unpairing volume ids %s and %s' % (self.parameters['src_vol_id'], self.parameters['dest_vol_id']), exception=to_native(err))