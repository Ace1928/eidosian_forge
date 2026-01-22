from __future__ import absolute_import, division, print_function
import time
class AzureRMPostgreSqlServers(AzureRMModuleBase):
    """Configuration class for an Azure RM PostgreSQL Server resource"""

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), sku=dict(type='dict'), location=dict(type='str'), storage_mb=dict(type='int'), geo_redundant_backup=dict(type='bool', default=False), backup_retention_days=dict(type='int'), version=dict(type='str', choices=['9.5', '9.6', '10', '11']), enforce_ssl=dict(type='bool', default=False), storage_autogrow=dict(type='bool', default=False), create_mode=dict(type='str', default='default', choices=['default', 'geo_restore', 'point_in_time_restore']), source_server_id=dict(type='str'), restore_point_in_time=dict(type='str'), admin_username=dict(type='str'), admin_password=dict(type='str', no_log=True), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.resource_group = None
        self.name = None
        self.parameters = {'properties': {'create_mode': 'default'}}
        self.tags = None
        self.results = dict(changed=False)
        self.state = None
        self.to_do = Actions.NoAction
        super(AzureRMPostgreSqlServers, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        """Main module execution method"""
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            elif kwargs[key] is not None:
                if key == 'sku':
                    ev = kwargs[key]
                    if 'tier' in ev:
                        if ev['tier'] == 'basic':
                            ev['tier'] = 'Basic'
                        elif ev['tier'] == 'standard':
                            ev['tier'] = 'Standard'
                    self.parameters['sku'] = ev
                elif key == 'location':
                    self.parameters['location'] = kwargs[key]
                elif key == 'storage_mb':
                    self.parameters.setdefault('properties', {}).setdefault('storage_profile', {})['storage_mb'] = kwargs[key]
                elif key == 'storage_autogrow':
                    self.parameters.setdefault('properties', {}).setdefault('storage_profile', {})['storage_autogrow'] = 'Enabled' if kwargs[key] else 'Disabled'
                elif key == 'geo_redundant_backup':
                    self.parameters.setdefault('properties', {}).setdefault('storage_profile', {})['geo_redundant_backup'] = 'Enabled' if kwargs[key] else 'Disabled'
                elif key == 'backup_retention_days':
                    self.parameters.setdefault('properties', {}).setdefault('storage_profile', {})['backup_retention_days'] = kwargs[key]
                elif key == 'version':
                    self.parameters.setdefault('properties', {})['version'] = kwargs[key]
                elif key == 'enforce_ssl':
                    self.parameters.setdefault('properties', {})['ssl_enforcement'] = 'Enabled' if kwargs[key] else 'Disabled'
                elif key == 'create_mode':
                    if kwargs[key] == 'default':
                        self.parameters['properties']['create_mode'] = 'Default'
                    elif kwargs[key] == 'point_in_time_restore':
                        self.parameters['properties']['create_mode'] = 'PointInTimeRestore'
                    elif kwargs[key] == 'geo_restore':
                        self.parameters['properties']['create_mode'] = 'GeoRestore'
                elif key == 'admin_username':
                    self.parameters.setdefault('properties', {})['administrator_login'] = kwargs[key]
                elif key == 'admin_password':
                    self.parameters.setdefault('properties', {})['administrator_login_password'] = kwargs[key]
                elif key == 'source_server_id':
                    self.parameters['properties']['source_server_id'] = kwargs[key]
                elif key == 'restore_point_in_time':
                    self.parameters['properties']['restore_point_in_time'] = kwargs[key]
        old_response = None
        response = None
        resource_group = self.get_resource_group(self.resource_group)
        if 'location' not in self.parameters:
            self.parameters['location'] = resource_group.location
        old_response = self.get_postgresqlserver()
        if not old_response:
            self.log("PostgreSQL Server instance doesn't exist")
            if self.state == 'absent':
                self.log("Old instance didn't exist")
            else:
                self.to_do = Actions.Create
        else:
            self.log('PostgreSQL Server instance already exists')
            if self.state == 'absent':
                self.to_do = Actions.Delete
            elif self.state == 'present':
                self.log('Need to check if PostgreSQL Server instance has to be deleted or may be updated')
                update_tags, newtags = self.update_tags(old_response.get('tags', {}))
                if update_tags:
                    self.tags = newtags
                self.to_do = Actions.Update
        if self.to_do == Actions.Create or self.to_do == Actions.Update:
            self.log('Need to Create / Update the PostgreSQL Server instance')
            if self.check_mode:
                self.results['changed'] = True
                return self.results
            response = self.create_update_postgresqlserver()
            if not old_response:
                self.results['changed'] = True
            else:
                self.results['changed'] = old_response.__ne__(response)
            self.log('Creation / Update done')
        elif self.to_do == Actions.Delete:
            self.log('PostgreSQL Server instance deleted')
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            self.delete_postgresqlserver()
            while self.get_postgresqlserver():
                time.sleep(20)
        else:
            self.log('PostgreSQL Server instance unchanged')
            self.results['changed'] = False
            response = old_response
        if response:
            self.results['id'] = response['id']
            self.results['version'] = response['version']
            self.results['state'] = response['user_visible_state']
            self.results['fully_qualified_domain_name'] = response['fully_qualified_domain_name']
        return self.results

    def create_update_postgresqlserver(self):
        """
        Creates or updates PostgreSQL Server with the specified configuration.

        :return: deserialized PostgreSQL Server instance state dictionary
        """
        self.log('Creating / Updating the PostgreSQL Server instance {0}'.format(self.name))
        try:
            self.parameters['tags'] = self.tags
            if self.to_do == Actions.Create:
                response = self.postgresql_client.servers.begin_create(resource_group_name=self.resource_group, server_name=self.name, parameters=self.parameters)
            else:
                self.parameters.update(self.parameters.pop('properties', {}))
                response = self.postgresql_client.servers.begin_update(resource_group_name=self.resource_group, server_name=self.name, parameters=self.parameters)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        except Exception as exc:
            self.log('Error attempting to create the PostgreSQL Server instance.')
            self.fail('Error creating the PostgreSQL Server instance: {0}'.format(str(exc)))
        return response.as_dict()

    def delete_postgresqlserver(self):
        """
        Deletes specified PostgreSQL Server instance in the specified subscription and resource group.

        :return: True
        """
        self.log('Deleting the PostgreSQL Server instance {0}'.format(self.name))
        try:
            response = self.postgresql_client.servers.begin_delete(resource_group_name=self.resource_group, server_name=self.name)
        except Exception as e:
            self.log('Error attempting to delete the PostgreSQL Server instance.')
            self.fail('Error deleting the PostgreSQL Server instance: {0}'.format(str(e)))
        return True

    def get_postgresqlserver(self):
        """
        Gets the properties of the specified PostgreSQL Server.

        :return: deserialized PostgreSQL Server instance state dictionary
        """
        self.log('Checking if the PostgreSQL Server instance {0} is present'.format(self.name))
        found = False
        try:
            response = self.postgresql_client.servers.get(resource_group_name=self.resource_group, server_name=self.name)
            found = True
            self.log('Response : {0}'.format(response))
            self.log('PostgreSQL Server instance : {0} found'.format(response.name))
        except ResourceNotFoundError as e:
            self.log('Did not find the PostgreSQL Server instance.')
        if found is True:
            return response.as_dict()
        return False