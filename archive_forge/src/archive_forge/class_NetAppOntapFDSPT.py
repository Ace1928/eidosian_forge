from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
class NetAppOntapFDSPT:
    """
        Creates, Modifies and removes a File Directory Security Policy Tasks
    """

    def __init__(self):
        """
            Initialize the Ontap File Directory Security Policy Tasks class
        """
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, choices=['present', 'absent'], default='present'), vserver=dict(required=True, type='str'), name=dict(required=True, type='str'), path=dict(required=True, type='str'), access_control=dict(required=False, choices=['file_directory', 'slag'], type='str'), ntfs_sd=dict(required=False, type='list', elements='str'), ntfs_mode=dict(required=False, choices=['propagate', 'ignore', 'replace'], type='str'), security_type=dict(required=False, choices=['ntfs', 'nfsv4'], type='str'), index_num=dict(required=False, type='int')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = OntapRestAPI(self.module)
        self.use_rest = self.rest_api.is_rest()
        if not self.use_rest:
            self.module.fail_json(msg=self.rest_api.requires_ontap_version('na_ontap_fdspt', '9.6'))

    def get_fdspt(self):
        """
        Get File Directory Security Policy Task
        """
        api = 'private/cli/vserver/security/file-directory/policy/task'
        query = {'policy-name': self.parameters['name'], 'path': self.parameters['path'], 'fields': 'vserver,ntfs-mode,ntfs-sd,security-type,access-control,index-num'}
        message, error = self.rest_api.get(api, query)
        records, error = rrh.check_for_0_or_1_records(api, message, error)
        if error:
            self.module.fail_json(msg=error)
        if records:
            if 'ntfs_sd' not in records:
                records['ntfs_sd'] = []
        return records if records else None

    def add_fdspt(self):
        """
        Adds a new File Directory Security Policy Task
        """
        api = 'private/cli/vserver/security/file-directory/policy/task/add'
        body = {'policy-name': self.parameters['name'], 'vserver': self.parameters['vserver'], 'path': self.parameters['path']}
        for i in ('ntfs_mode', 'ntfs_sd', 'security_type', 'access_control', 'index_num'):
            if i in self.parameters:
                body[i.replace('_', '-')] = self.parameters[i]
        dummy, error = self.rest_api.post(api, body)
        if error:
            self.module.fail_json(msg=error)

    def remove_fdspt(self):
        """
        Deletes a File Directory Security Policy Task
        """
        api = 'private/cli/vserver/security/file-directory/policy/task/remove'
        body = {'policy-name': self.parameters['name'], 'vserver': self.parameters['vserver'], 'path': self.parameters['path']}
        dummy, error = self.rest_api.delete(api, body)
        if error:
            self.module.fail_json(msg=error)

    def modify_fdspt(self):
        """
        Modifies a File Directory Security Policy Task
        """
        self.remove_fdspt()
        self.add_fdspt()

    def apply(self):
        current, modify = (self.get_fdspt(), None)
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        if cd_action is None and self.parameters['state'] == 'present':
            modify = self.na_helper.get_modified_attributes(current, self.parameters)
        if self.na_helper.changed and (not self.module.check_mode):
            if cd_action == 'create':
                self.add_fdspt()
            elif cd_action == 'delete':
                self.remove_fdspt()
            elif modify:
                self.modify_fdspt()
        result = netapp_utils.generate_result(self.na_helper.changed, cd_action, modify)
        self.module.exit_json(**result)