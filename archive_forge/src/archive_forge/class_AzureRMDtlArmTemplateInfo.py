from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMDtlArmTemplateInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), lab_name=dict(type='str', required=True), artifact_source_name=dict(type='str', required=True), name=dict(type='str'))
        self.results = dict(changed=False)
        self.mgmt_client = None
        self.resource_group = None
        self.lab_name = None
        self.artifact_source_name = None
        self.name = None
        super(AzureRMDtlArmTemplateInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_devtestlabarmtemplate_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_devtestlabarmtemplate_facts' module has been renamed to 'azure_rm_devtestlabarmtemplate_info'", version=(2.9,))
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        self.mgmt_client = self.get_mgmt_svc_client(DevTestLabsClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        if self.name:
            self.results['armtemplates'] = self.get()
        else:
            self.results['armtemplates'] = self.list()
        return self.results

    def list(self):
        response = None
        results = []
        try:
            response = self.mgmt_client.arm_templates.list(resource_group_name=self.resource_group, lab_name=self.lab_name, artifact_source_name=self.artifact_source_name)
            self.log('Response : {0}'.format(response))
        except Exception as e:
            self.fail('Could not get facts for DTL ARM Template.')
        if response is not None:
            for item in response:
                results.append(self.format_response(item))
        return results

    def get(self):
        response = None
        results = []
        try:
            response = self.mgmt_client.arm_templates.get(resource_group_name=self.resource_group, lab_name=self.lab_name, artifact_source_name=self.artifact_source_name, name=self.name)
            self.log('Response : {0}'.format(response))
        except ResourceNotFoundError as e:
            self.fail('Could not get facts for DTL ARM Template.')
        if response:
            results.append(self.format_response(response))
        return results

    def format_response(self, item):
        d = item.as_dict()
        d = {'resource_group': self.parse_resource_to_dict(d.get('id')).get('resource_group'), 'lab_name': self.parse_resource_to_dict(d.get('id')).get('name'), 'artifact_source_name': self.parse_resource_to_dict(d.get('id')).get('child_name_1'), 'id': d.get('id', None), 'name': d.get('name'), 'display_name': d.get('display_name'), 'description': d.get('description'), 'publisher': d.get('publisher')}
        return d