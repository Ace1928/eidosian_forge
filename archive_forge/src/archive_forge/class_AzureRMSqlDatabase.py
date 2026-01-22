from __future__ import absolute_import, division, print_function
import time
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase, format_resource_id
class AzureRMSqlDatabase(AzureRMModuleBase):
    """Configuration class for an Azure RM SQL Database resource"""

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), server_name=dict(type='str', required=True), name=dict(type='str', required=True), location=dict(type='str'), collation=dict(type='str'), create_mode=dict(type='str', choices=['copy', 'default', 'non_readable_secondary', 'online_secondary', 'point_in_time_restore', 'recovery', 'restore', 'restore_long_term_retention_backup']), source_database_id=dict(type='str'), source_database_deletion_date=dict(type='str'), restore_point_in_time=dict(type='str'), recovery_services_recovery_point_resource_id=dict(type='str'), edition=dict(type='str', choices=['web', 'business', 'basic', 'standard', 'premium', 'free', 'stretch', 'data_warehouse', 'system', 'system2']), sku=dict(type='dict', options=sku_spec), max_size_bytes=dict(type='str'), elastic_pool_name=dict(type='str'), read_scale=dict(type='bool', default=False), sample_name=dict(type='str', choices=['adventure_works_lt']), zone_redundant=dict(type='bool', default=False), force_update=dict(type='bool'), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.resource_group = None
        self.server_name = None
        self.name = None
        self.parameters = dict()
        self.tags = None
        self.results = dict(changed=False)
        self.state = None
        self.to_do = Actions.NoAction
        super(AzureRMSqlDatabase, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        """Main module execution method"""
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            elif kwargs[key] is not None:
                if key == 'location':
                    self.parameters['location'] = kwargs[key]
                elif key == 'collation':
                    self.parameters['collation'] = kwargs[key]
                elif key == 'create_mode':
                    self.parameters['create_mode'] = _snake_to_camel(kwargs[key], True)
                elif key == 'source_database_id':
                    self.parameters['source_database_id'] = kwargs[key]
                elif key == 'source_database_deletion_date':
                    try:
                        self.parameters['source_database_deletion_date'] = dateutil.parser.parse(kwargs[key])
                    except dateutil.parser._parser.ParserError:
                        self.fail('Error parsing date from source_database_deletion_date: {0}'.format(kwargs[key]))
                elif key == 'restore_point_in_time':
                    try:
                        self.parameters['restore_point_in_time'] = dateutil.parser.parse(kwargs[key])
                    except dateutil.parser._parser.ParserError:
                        self.fail('Error parsing date from restore_point_in_time: {0}'.format(kwargs[key]))
                elif key == 'recovery_services_recovery_point_resource_id':
                    self.parameters['recovery_services_recovery_point_resource_id'] = kwargs[key]
                elif key == 'edition':
                    ev = get_sku_name(kwargs[key])
                    self.parameters['sku'] = Sku(name=ev)
                elif key == 'sku':
                    ev = kwargs[key]
                    self.parameters['sku'] = Sku(name=ev['name'], tier=ev['tier'], size=ev['size'], family=ev['family'], capacity=ev['capacity'])
                elif key == 'max_size_bytes':
                    self.parameters['max_size_bytes'] = kwargs[key]
                elif key == 'elastic_pool_name':
                    self.parameters['elastic_pool_id'] = kwargs[key]
                elif key == 'read_scale':
                    self.parameters['read_scale'] = 'Enabled' if kwargs[key] else 'Disabled'
                elif key == 'sample_name':
                    ev = kwargs[key]
                    if ev == 'adventure_works_lt':
                        ev = 'AdventureWorksLT'
                    self.parameters['sample_name'] = ev
                elif key == 'zone_redundant':
                    self.parameters['zone_redundant'] = True if kwargs[key] else False
        old_response = None
        response = None
        resource_group = self.get_resource_group(self.resource_group)
        if 'location' not in self.parameters:
            self.parameters['location'] = resource_group.location
        if 'elastic_pool_id' in self.parameters:
            self.format_elastic_pool_id()
        old_response = self.get_sqldatabase()
        if not old_response:
            self.log("SQL Database instance doesn't exist")
            if self.state == 'absent':
                self.log("Old instance didn't exist")
            else:
                self.to_do = Actions.Create
        else:
            self.log('SQL Database instance already exists')
            if self.state == 'absent':
                self.to_do = Actions.Delete
            elif self.state == 'present':
                self.log('Need to check if SQL Database instance has to be deleted or may be updated')
                if 'location' in self.parameters and self.parameters['location'] != old_response['location']:
                    self.to_do = Actions.Update
                if 'read_scale' in self.parameters and self.parameters['read_scale'] != old_response['read_scale']:
                    self.to_do = Actions.Update
                if 'max_size_bytes' in self.parameters and self.parameters['max_size_bytes'] != old_response['max_size_bytes']:
                    self.to_do = Actions.Update
                if 'sku' in self.parameters and self.parameters['sku'].as_dict() != old_response['sku']:
                    self.to_do = Actions.Update
                update_tags, newtags = self.update_tags(old_response.get('tags', dict()))
                if update_tags:
                    self.tags = newtags
                    self.to_do = Actions.Update
        if self.to_do == Actions.Create or self.to_do == Actions.Update:
            self.log('Need to Create / Update the SQL Database instance')
            if self.check_mode:
                self.results['changed'] = True
                return self.results
            self.parameters['tags'] = self.tags
            response = self.create_update_sqldatabase()
            if not old_response:
                self.results['changed'] = True
            else:
                self.results['changed'] = old_response.__ne__(response)
            self.log('Creation / Update done')
        elif self.to_do == Actions.Delete:
            self.log('SQL Database instance deleted')
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            self.delete_sqldatabase()
            while self.get_sqldatabase():
                time.sleep(20)
        else:
            self.log('SQL Database instance unchanged')
            self.results['changed'] = False
            response = old_response
        if response:
            self.results['id'] = response['id']
            self.results['database_id'] = response['database_id']
            self.results['status'] = response['status']
        return self.results

    def create_update_sqldatabase(self):
        """
        Creates or updates SQL Database with the specified configuration.

        :return: deserialized SQL Database instance state dictionary
        """
        self.log('Creating / Updating the SQL Database instance {0}'.format(self.name))
        try:
            response = self.sql_client.databases.begin_create_or_update(resource_group_name=self.resource_group, server_name=self.server_name, database_name=self.name, parameters=self.parameters)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        except Exception as exc:
            self.log('Error attempting to create the SQL Database instance.')
            self.fail('Error creating the SQL Database instance: {0}'.format(str(exc)))
        return response.as_dict()

    def delete_sqldatabase(self):
        """
        Deletes specified SQL Database instance in the specified subscription and resource group.

        :return: True
        """
        self.log('Deleting the SQL Database instance {0}'.format(self.name))
        try:
            response = self.sql_client.databases.begin_delete(resource_group_name=self.resource_group, server_name=self.server_name, database_name=self.name)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
        except Exception as e:
            self.log('Error attempting to delete the SQL Database instance.')
            self.fail('Error deleting the SQL Database instance: {0}'.format(str(e)))
        return True

    def get_sqldatabase(self):
        """
        Gets the properties of the specified SQL Database.

        :return: deserialized SQL Database instance state dictionary
        """
        self.log('Checking if the SQL Database instance {0} is present'.format(self.name))
        found = False
        try:
            response = self.sql_client.databases.get(resource_group_name=self.resource_group, server_name=self.server_name, database_name=self.name)
            found = True
            self.log('Response : {0}'.format(response))
            self.log('SQL Database instance : {0} found'.format(response.name))
        except ResourceNotFoundError:
            self.log('Did not find the SQL Database instance.')
        if found is True:
            return response.as_dict()
        return False

    def format_elastic_pool_id(self):
        parrent_id = format_resource_id(val=self.server_name, subscription_id=self.subscription_id, namespace='Microsoft.Sql', types='servers', resource_group=self.resource_group)
        self.parameters['elastic_pool_id'] = parrent_id + '/elasticPools/' + self.parameters['elastic_pool_id']