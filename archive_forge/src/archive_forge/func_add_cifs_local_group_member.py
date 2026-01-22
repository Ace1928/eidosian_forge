from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
def add_cifs_local_group_member(self):
    """
        Adds a member to a CIFS local group
        """
    if self.use_rest:
        api = 'protocols/cifs/local-groups/%s/%s/members' % (self.svm_uuid, self.sid)
        body = {'name': self.parameters['member']}
        dummy, error = rest_generic.post_async(self.rest_api, api, body)
        if error:
            self.module.fail_json(msg='Error adding member %s to cifs local group %s on vserver %s: %s' % (self.parameters['member'], self.parameters['group'], self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())
    else:
        group_members_obj = netapp_utils.zapi.NaElement('cifs-local-group-members-add-members')
        group_members_obj.add_new_child('group-name', self.parameters['group'])
        member_names = netapp_utils.zapi.NaElement('member-names')
        member_names.add_new_child('cifs-name', self.parameters['member'])
        group_members_obj.add_child_elem(member_names)
        try:
            self.server.invoke_successfully(group_members_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            self.module.fail_json(msg='Error adding member %s to cifs local group %s on vserver %s: %s' % (self.parameters['member'], self.parameters['group'], self.parameters['vserver'], to_native(error)), exception=traceback.format_exc())