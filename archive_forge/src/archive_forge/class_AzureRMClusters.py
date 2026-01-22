from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMClusters(AzureRMModuleBase):
    """Configuration class for an Azure RM Cluster resource"""

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), location=dict(type='str'), cluster_version=dict(type='str'), os_type=dict(type='str', choices=['linux']), tier=dict(type='str', choices=['standard', 'premium']), cluster_definition=dict(type='dict'), compute_profile_roles=dict(type='list', elements='dict', options=dict(name=dict(type='str', choices=['headnode', 'workernode', 'zookepernode']), min_instance_count=dict(type='int'), target_instance_count=dict(type='int'), vm_size=dict(type='str'), linux_profile=dict(type='dict', options=dict(username=dict(type='str'), password=dict(type='str', no_log=True))))), storage_accounts=dict(type='list', elements='dict', options=dict(name=dict(type='str'), is_default=dict(type='bool'), container=dict(type='str'), key=dict(type='str', no_log=True))), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.resource_group = None
        self.name = None
        self.parameters = dict()
        self.results = dict(changed=False)
        self.mgmt_client = None
        self.state = None
        self.to_do = Actions.NoAction
        self.tags_changed = False
        self.new_instance_count = None
        super(AzureRMClusters, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        """Main module execution method"""
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            elif kwargs[key] is not None:
                self.parameters[key] = kwargs[key]
        dict_expand(self.parameters, ['cluster_version'], 'properties')
        dict_camelize(self.parameters, ['os_type'], True)
        dict_expand(self.parameters, ['os_type'], 'properties')
        dict_camelize(self.parameters, ['tier'], True)
        dict_expand(self.parameters, ['tier'], 'properties')
        dict_rename(self.parameters, ['cluster_definition', 'gateway_rest_username'], 'restAuthCredential.username')
        dict_rename(self.parameters, ['cluster_definition', 'gateway_rest_password'], 'restAuthCredential.password')
        dict_expand(self.parameters, ['cluster_definition', 'restAuthCredential.username'], 'gateway')
        dict_expand(self.parameters, ['cluster_definition', 'restAuthCredential.password'], 'gateway')
        dict_expand(self.parameters, ['cluster_definition', 'gateway'], 'configurations')
        dict_expand(self.parameters, ['cluster_definition'], 'properties')
        dict_expand(self.parameters, ['compute_profile_roles', 'vm_size'], 'hardware_profile')
        dict_rename(self.parameters, ['compute_profile_roles', 'linux_profile'], 'linux_operating_system_profile')
        dict_expand(self.parameters, ['compute_profile_roles', 'linux_operating_system_profile'], 'os_profile')
        dict_rename(self.parameters, ['compute_profile_roles'], 'roles')
        dict_expand(self.parameters, ['roles'], 'compute_profile')
        dict_expand(self.parameters, ['compute_profile'], 'properties')
        dict_rename(self.parameters, ['storage_accounts'], 'storageaccounts')
        dict_expand(self.parameters, ['storageaccounts'], 'storage_profile')
        dict_expand(self.parameters, ['storage_profile'], 'properties')
        response = None
        self.mgmt_client = self.get_mgmt_svc_client(HDInsightManagementClient, is_track2=True, base_url=self._cloud_environment.endpoints.resource_manager)
        resource_group = self.get_resource_group(self.resource_group)
        if 'location' not in self.parameters:
            self.parameters['location'] = resource_group.location
        old_response = self.get_cluster()
        if not old_response:
            self.log("Cluster instance doesn't exist")
            if self.state == 'absent':
                self.log("Old instance didn't exist")
            else:
                self.to_do = Actions.Create
        else:
            self.log('Cluster instance already exists')
            if self.state == 'absent':
                self.to_do = Actions.Delete
            elif self.state == 'present':
                compare_result = {}
                if not default_compare(self.parameters, old_response, '', compare_result):
                    if compare_result.pop('/properties/compute_profile/roles/*/target_instance_count', False):
                        new_count = 0
                        old_count = 0
                        for role in self.parameters['properties']['compute_profile']['roles']:
                            if role['name'] == 'workernode':
                                new_count = role['target_instance_count']
                        for role in old_response['properties']['compute_profile']['roles']:
                            if role['name'] == 'workernode':
                                old_count = role['target_instance_count']
                        if old_count != new_count:
                            self.new_instance_count = new_count
                            self.to_do = Actions.Update
                    if compare_result.pop('/tags', False):
                        self.to_do = Actions.Update
                        self.tags_changed = True
                    if compare_result:
                        for k in compare_result.keys():
                            self.module.warn("property '" + k + "' cannot be updated (" + compare_result[k] + ')')
                        self.module.warn('only tags and target_instance_count can be updated')
        if self.to_do == Actions.Create or self.to_do == Actions.Update:
            self.log('Need to Create / Update the Cluster instance')
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            response = self.create_update_cluster()
            self.log('Creation / Update done')
        elif self.to_do == Actions.Delete:
            self.log('Cluster instance deleted')
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            self.delete_cluster()
        else:
            self.log('Cluster instance unchanged')
            self.results['changed'] = False
            response = old_response
        if self.state == 'present':
            self.results.update(self.format_item(response))
        return self.results

    def create_update_cluster(self):
        """
        Creates or updates Cluster with the specified configuration.

        :return: deserialized Cluster instance state dictionary
        """
        self.log('Creating / Updating the Cluster instance {0}'.format(self.name))
        try:
            if self.to_do == Actions.Create:
                response = self.mgmt_client.clusters.begin_create(resource_group_name=self.resource_group, cluster_name=self.name, parameters=self.parameters)
                if isinstance(response, LROPoller):
                    response = self.get_poller_result(response)
            else:
                if self.tags_changed:
                    response = self.mgmt_client.clusters.update(resource_group_name=self.resource_group, cluster_name=self.name, parameters={'tags': self.parameters.get('tags')})
                    if isinstance(response, LROPoller):
                        response = self.get_poller_result(response)
                if self.new_instance_count:
                    response = self.mgmt_client.clusters.begin_resize(resource_group_name=self.resource_group, cluster_name=self.name, role_name='workernode', parameters={'target_instance_count': self.new_instance_count})
                    if isinstance(response, LROPoller):
                        response = self.get_poller_result(response)
        except Exception as exc:
            self.fail('Error creating or updating Cluster instance: {0}'.format(str(exc)))
        return response.as_dict() if response else {}

    def delete_cluster(self):
        """
        Deletes specified Cluster instance in the specified subscription and resource group.

        :return: True
        """
        self.log('Deleting the Cluster instance {0}'.format(self.name))
        try:
            response = self.mgmt_client.clusters.begin_delete(resource_group_name=self.resource_group, cluster_name=self.name)
        except Exception as e:
            self.fail('Error deleting the Cluster instance: {0}'.format(str(e)))
        return True

    def get_cluster(self):
        """
        Gets the properties of the specified Cluster.

        :return: deserialized Cluster instance state dictionary
        """
        self.log('Checking if the Cluster instance {0} is present'.format(self.name))
        found = False
        try:
            response = self.mgmt_client.clusters.get(resource_group_name=self.resource_group, cluster_name=self.name)
            found = True
            self.log('Response : {0}'.format(response))
            self.log('Cluster instance : {0} found'.format(response.name))
        except ResourceNotFoundError as e:
            self.log('Did not find the Cluster instance.')
        if found is True:
            return response.as_dict()
        return False

    def format_item(self, d):
        d = {'id': d.get('id', None)}
        return d