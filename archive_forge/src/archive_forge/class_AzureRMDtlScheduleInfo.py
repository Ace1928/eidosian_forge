from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _camel_to_snake, _snake_to_camel
class AzureRMDtlScheduleInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), lab_name=dict(type='str', required=True), name=dict(type='str'), tags=dict(type='list', elements='str'))
        self.results = dict(changed=False)
        self.mgmt_client = None
        self.resource_group = None
        self.lab_name = None
        self.name = None
        self.tags = None
        super(AzureRMDtlScheduleInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_devtestlabschedule_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_devtestlabschedule_facts' module has been renamed to 'azure_rm_devtestlabschedule_info'", version=(2.9,))
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        self.mgmt_client = self.get_mgmt_svc_client(DevTestLabsClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        if self.name:
            self.results['schedules'] = self.get()
        else:
            self.results['schedules'] = self.list()
        return self.results

    def get(self):
        response = None
        results = []
        try:
            response = self.mgmt_client.schedules.get(resource_group_name=self.resource_group, lab_name=self.lab_name, name=_snake_to_camel(self.name))
            self.log('Response : {0}'.format(response))
        except ResourceNotFoundError as e:
            self.log('Could not get facts for Schedule.')
        if response and self.has_tags(response.tags, self.tags):
            results.append(self.format_response(response))
        return results

    def list(self):
        response = None
        results = []
        try:
            response = self.mgmt_client.schedules.list(resource_group_name=self.resource_group, lab_name=self.lab_name)
            self.log('Response : {0}'.format(response))
        except Exception as e:
            self.log('Could not get facts for Schedule.')
        if response is not None:
            for item in response:
                if self.has_tags(item.tags, self.tags):
                    results.append(self.format_response(item))
        return results

    def format_response(self, item):
        d = item.as_dict()
        d = {'resource_group': self.resource_group, 'lab_name': self.lab_name, 'name': _camel_to_snake(d.get('name')), 'id': d.get('id', None), 'tags': d.get('tags', None), 'time': d.get('daily_recurrence', {}).get('time'), 'time_zone_id': d.get('time_zone_id')}
        return d