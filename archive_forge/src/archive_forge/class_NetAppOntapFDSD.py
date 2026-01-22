from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
class NetAppOntapFDSD:
    """
        Creates or removes a File Directory Security Descriptor
    """

    def __init__(self):
        """
            Initialize the ONTAP File Directory Security Descripter class
        """
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, choices=['present', 'absent'], default='present'), vserver=dict(required=True, type='str'), name=dict(required=True, type='str')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = OntapRestAPI(self.module)
        self.use_rest = self.rest_api.is_rest()
        if not self.use_rest:
            self.module.fail_json(msg=self.rest_api.requires_ontap_version('na_ontap_fdsd', '9.6'))

    def get_fdsd(self):
        """
        Get File Directory Security Descriptor
        """
        api = 'private/cli/vserver/security/file-directory/ntfs'
        query = {'ntfs-sd': self.parameters['name'], 'vserver': self.parameters['vserver']}
        message, error = self.rest_api.get(api, query)
        records, error = rrh.check_for_0_or_more_records(api, message, error)
        if error:
            self.module.fail_json(msg=error)
        return records if records else None

    def add_fdsd(self):
        """
        Adds a new File Directory Security Descriptor
        """
        api = 'private/cli/vserver/security/file-directory/ntfs'
        body = {'ntfs-sd': self.parameters['name'], 'vserver': self.parameters['vserver']}
        dummy, error = self.rest_api.post(api, body)
        if error:
            self.module.fail_json(msg=error)

    def remove_fdsd(self):
        """
        Deletes a File Directory Security Descriptor
        """
        api = 'private/cli/vserver/security/file-directory/ntfs'
        body = {'ntfs-sd': self.parameters['name'], 'vserver': self.parameters['vserver']}
        dummy, error = self.rest_api.delete(api, body)
        if error:
            self.module.fail_json(msg=error)

    def apply(self):
        current = self.get_fdsd()
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        if self.na_helper.changed:
            if not self.module.check_mode:
                if cd_action == 'create':
                    self.add_fdsd()
                elif cd_action == 'delete':
                    self.remove_fdsd()
        result = netapp_utils.generate_result(self.na_helper.changed, cd_action)
        self.module.exit_json(**result)