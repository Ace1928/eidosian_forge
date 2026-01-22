from __future__ import absolute_import, division, print_function
from ansible.module_utils.common.dict_transformations import _camel_to_snake
from ansible.module_utils._text import to_native
from datetime import datetime, timedelta
class AzureRMServiceBusInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(name=dict(type='str'), resource_group=dict(type='str'), tags=dict(type='list', elements='str'), type=dict(type='str', required=True, choices=['namespace', 'topic', 'queue', 'subscription']), namespace=dict(type='str'), topic=dict(type='str'), show_sas_policies=dict(type='bool'))
        required_if = [('type', 'subscription', ['topic', 'resource_group', 'namespace']), ('type', 'topic', ['resource_group', 'namespace']), ('type', 'queue', ['resource_group', 'namespace'])]
        self.results = dict(changed=False, servicebuses=[])
        self.name = None
        self.resource_group = None
        self.tags = None
        self.type = None
        self.namespace = None
        self.topic = None
        self.show_sas_policies = None
        super(AzureRMServiceBusInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, required_if=required_if, facts_module=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_servicebus_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_servicebus_facts' module has been renamed to 'azure_rm_servicebus_info'", version=(2.9,))
        for key in self.module_arg_spec:
            setattr(self, key, kwargs[key])
        response = []
        if self.name:
            response = self.get_item()
        elif self.resource_group:
            response = self.list_items()
        else:
            response = self.list_all_items()
        self.results['servicebuses'] = [self.instance_to_dict(x) for x in response]
        return self.results

    def instance_to_dict(self, instance):
        result = dict()
        instance_type = getattr(self.servicebus_models, 'SB{0}'.format(str.capitalize(self.type)))
        attribute_map = instance_type._attribute_map
        for attribute in attribute_map.keys():
            value = getattr(instance, attribute)
            if attribute_map[attribute]['type'] == 'duration':
                if is_valid_timedelta(value):
                    key = duration_spec_map.get(attribute) or attribute
                    result[key] = int(value.total_seconds())
            elif attribute == 'status':
                result['status'] = _camel_to_snake(value)
            elif isinstance(value, self.servicebus_models.MessageCountDetails):
                result[attribute] = value.as_dict()
            elif isinstance(value, self.servicebus_models.SBSku):
                result[attribute] = value.name.lower()
            elif isinstance(value, datetime):
                result[attribute] = str(value)
            elif isinstance(value, str):
                result[attribute] = to_native(value)
            elif attribute == 'max_size_in_megabytes':
                result['max_size_in_mb'] = value
            else:
                result[attribute] = value
        if self.show_sas_policies and self.type != 'subscription':
            policies = self.get_auth_rules()
            for name in policies.keys():
                policies[name]['keys'] = self.get_sas_key(name)
            result['sas_policies'] = policies
        if self.namespace:
            result['namespace'] = self.namespace
        if self.topic:
            result['topic'] = self.topic
        return result

    def _get_client(self):
        return getattr(self.servicebus_client, '{0}s'.format(self.type))

    def get_item(self):
        try:
            client = self._get_client()
            if self.type == 'namespace':
                item = client.get(self.resource_group, self.name)
                return [item] if self.has_tags(item.tags, self.tags) else []
            elif self.type == 'subscription':
                return [client.get(self.resource_group, self.namespace, self.topic, self.name)]
            else:
                return [client.get(self.resource_group, self.namespace, self.name)]
        except Exception:
            pass
        return []

    def list_items(self):
        try:
            client = self._get_client()
            if self.type == 'namespace':
                response = client.list_by_resource_group(self.resource_group)
                return [x for x in response if self.has_tags(x.tags, self.tags)]
            elif self.type == 'subscription':
                return client.list_by_topic(self.resource_group, self.namespace, self.topic)
            else:
                return client.list_by_namespace(self.resource_group, self.namespace)
        except Exception as exc:
            self.fail('Failed to list items - {0}'.format(str(exc)))
        return []

    def list_all_items(self):
        self.log('List all items in subscription')
        try:
            if self.type != 'namespace':
                return []
            response = self.servicebus_client.namespaces.list()
            return [x for x in response if self.has_tags(x.tags, self.tags)]
        except Exception as exc:
            self.fail('Failed to list all items - {0}'.format(str(exc)))
        return []

    def get_auth_rules(self):
        result = dict()
        try:
            client = self._get_client()
            if self.type == 'namespace':
                rules = client.list_authorization_rules(self.resource_group, self.name)
            else:
                rules = client.list_authorization_rules(self.resource_group, self.namespace, self.name)
            while True:
                rule = rules.next()
                result[rule.name] = self.policy_to_dict(rule)
        except StopIteration:
            pass
        except Exception as exc:
            self.fail('Error when getting SAS policies for {0} {1}: {2}'.format(self.type, self.name, exc.message or str(exc)))
        return result

    def get_sas_key(self, name):
        try:
            client = self._get_client()
            if self.type == 'namespace':
                return client.list_keys(self.resource_group, self.name, name).as_dict()
            else:
                return client.list_keys(self.resource_group, self.namespace, self.name, name).as_dict()
        except Exception as exc:
            self.fail("Error when getting SAS policy {0}'s key - {1}".format(name, exc.message or str(exc)))
        return None

    def policy_to_dict(self, rule):
        result = rule.as_dict()
        rights = result['rights']
        if 'Manage' in rights:
            result['rights'] = 'manage'
        elif 'Listen' in rights and 'Send' in rights:
            result['rights'] = 'listen_send'
        else:
            result['rights'] = rights[0].lower()
        return result