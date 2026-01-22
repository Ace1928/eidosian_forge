from __future__ import absolute_import, division, print_function
import traceback
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
def add_ntfs_sd(self):
    """
        Adds a new NTFS security descriptor
        """
    ntfs_sd_obj = netapp_utils.zapi.NaElement('file-directory-security-ntfs-create')
    ntfs_sd_obj.add_new_child('ntfs-sd', self.parameters['name'])
    if self.parameters.get('control_flags_raw') is not None:
        ntfs_sd_obj.add_new_child('control-flags-raw', str(self.parameters['control_flags_raw']))
    if self.parameters.get('owner'):
        ntfs_sd_obj.add_new_child('owner', self.parameters['owner'])
    if self.parameters.get('group'):
        ntfs_sd_obj.add_new_child('group', self.parameters['group'])
    try:
        self.server.invoke_successfully(ntfs_sd_obj, True)
    except netapp_utils.zapi.NaApiError as error:
        self.module.fail_json(msg='Error creating NTFS security descriptor %s: %s' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())