from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
import time
class AzureRMEventHub(AzureRMModuleBase):

    def __init__(self):
        self.authorizations_spec = dict(name=dict(type='str', required=True))
        self.module_arg_spec = dict(message_retention_in_days=dict(type='int'), name=dict(type='str'), namespace_name=dict(type='str', required=True), partition_count=dict(type='int'), resource_group=dict(type='str', required=True), sku=dict(type='str', choices=['Basic', 'Standard'], default='Basic'), status=dict(choices=['Active', 'Disabled', 'Restoring', 'SendDisabled', 'ReceiveDisabled', 'Creating', 'Deleting', 'Renaming', 'Unknown'], default='Active', type='str'), state=dict(choices=['present', 'absent'], default='present', type='str'), location=dict(type='str'))
        required_if = [('state', 'present', ['partition_count', 'message_retention_in_days'])]
        self.sku = None
        self.resource_group = None
        self.namespace_name = None
        self.message_retention_in_days = None
        self.name = None
        self.location = None
        self.authorizations = None
        self.tags = None
        self.status = None
        self.partition_count = None
        self.results = dict(changed=False, state=dict())
        self.state = None
        super(AzureRMEventHub, self).__init__(derived_arg_spec=self.module_arg_spec, supports_check_mode=True, supports_tags=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            setattr(self, key, kwargs[key])
        resource_group = self.get_resource_group(self.resource_group)
        if not self.location:
            self.location = resource_group.location
        results = dict()
        changed = False
        try:
            self.log('Fetching Event Hub Namespace {0}'.format(self.name))
            namespace = self.event_hub_client.namespaces.get(self.resource_group, self.namespace_name)
            results = namespace_to_dict(namespace)
            event_hub_results = None
            if self.name:
                self.log('Fetching event Hub {0}'.format(self.name))
                event_hub = self.event_hub_client.event_hubs.get(self.resource_group, self.namespace_name, self.name)
                event_hub_results = event_hub_to_dict(event_hub)
            if self.state == 'present':
                changed = False
                update_tags, results['tags'] = self.update_tags(results['tags'])
                if update_tags:
                    changed = True
                elif self.namespace_name and (not self.name):
                    if self.sku != results['sku']:
                        changed = True
                elif self.namespace_name and self.name and event_hub_results:
                    if results['sku'] != 'Basic' and self.message_retention_in_days != event_hub_results['message_retention_in_days']:
                        self.sku = results['sku']
                        changed = True
            elif self.state == 'absent':
                changed = True
        except Exception:
            if self.state == 'present':
                changed = True
            else:
                changed = False
        self.results['changed'] = changed
        if self.name and (not changed):
            self.results['state'] = event_hub_results
        else:
            self.results['state'] = results
        if self.check_mode:
            return self.results
        if changed:
            if self.state == 'present':
                if self.name is None:
                    self.results['state'] = self.create_or_update_namespaces()
                elif self.namespace_name and self.name:
                    self.results['state'] = self.create_or_update_event_hub()
            elif self.state == 'absent':
                if self.name is None:
                    self.delete_namespace()
                elif self.namespace_name and self.name:
                    self.delete_event_hub()
                self.results['state']['status'] = 'Deleted'
        return self.results

    def create_or_update_namespaces(self):
        """
        create or update namespaces
        """
        try:
            namespace_params = EHNamespace(location=self.location, sku=Sku(name=self.sku), tags=self.tags)
            result = self.event_hub_client.namespaces.begin_create_or_update(self.resource_group, self.namespace_name, namespace_params)
            namespace = self.event_hub_client.namespaces.get(self.resource_group, self.namespace_name)
            while namespace.provisioning_state == 'Created':
                time.sleep(30)
                namespace = self.event_hub_client.namespaces.get(self.resource_group, self.namespace_name)
        except Exception as ex:
            self.fail('Failed to create namespace {0} in resource group {1}: {2}'.format(self.namespace_name, self.resource_group, str(ex)))
        return namespace_to_dict(namespace)

    def create_or_update_event_hub(self):
        """
        Create or update Event Hub.
        :return: create or update Event Hub instance state dictionary
        """
        try:
            if self.sku == 'Basic':
                self.message_retention_in_days = 1
            params = Eventhub(message_retention_in_days=self.message_retention_in_days, partition_count=self.partition_count, status=self.status)
            result = self.event_hub_client.event_hubs.create_or_update(self.resource_group, self.namespace_name, self.name, params)
            self.log('Response : {0}'.format(result))
        except Exception as ex:
            self.fail('Failed to create event hub {0} in resource group {1}: {2}'.format(self.name, self.resource_group, str(ex)))
        return event_hub_to_dict(result)

    def delete_event_hub(self):
        """
        Deletes specified event hub
        :return True
        """
        self.log('Deleting the event hub {0}'.format(self.name))
        try:
            result = self.event_hub_client.event_hubs.delete(self.resource_group, self.namespace_name, self.name)
        except Exception as e:
            self.log('Error attempting to delete event hub.')
            self.fail('Error deleting the event hub : {0}'.format(str(e)))
        return True

    def delete_namespace(self):
        """
        Deletes specified namespace
        :return True
        """
        self.log('Deleting the namespace {0}'.format(self.namespace_name))
        try:
            result = self.event_hub_client.namespaces.begin_delete(self.resource_group, self.namespace_name)
        except Exception as e:
            self.log('Error attempting to delete namespace.')
            self.fail('Error deleting the namespace : {0}'.format(str(e)))
        return True