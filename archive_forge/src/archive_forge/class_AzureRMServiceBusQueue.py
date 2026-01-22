from __future__ import absolute_import, division, print_function
from ansible.module_utils.common.dict_transformations import _snake_to_camel, _camel_to_snake
from ansible.module_utils._text import to_native
from datetime import datetime, timedelta
class AzureRMServiceBusQueue(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), state=dict(type='str', default='present', choices=['present', 'absent']), namespace=dict(type='str', required=True), auto_delete_on_idle_in_seconds=dict(type='int'), dead_lettering_on_message_expiration=dict(type='bool'), default_message_time_to_live_seconds=dict(type='int'), duplicate_detection_time_in_seconds=dict(type='int'), enable_batched_operations=dict(type='bool'), enable_express=dict(type='bool'), enable_partitioning=dict(type='bool'), forward_dead_lettered_messages_to=dict(type='str'), forward_to=dict(type='str'), lock_duration_in_seconds=dict(type='int'), max_delivery_count=dict(type='int'), max_message_size_in_kb=dict(type='int'), max_size_in_mb=dict(type='int'), requires_duplicate_detection=dict(type='bool'), requires_session=dict(type='bool'), status=dict(type='str', choices=['active', 'disabled', 'send_disabled', 'receive_disabled']))
        self.resource_group = None
        self.name = None
        self.state = None
        self.namespace = None
        self.location = None
        self.type = None
        self.subscription_topic_name = None
        self.auto_delete_on_idle_in_seconds = None
        self.dead_lettering_on_message_expiration = None
        self.default_message_time_to_live_seconds = None
        self.enable_batched_operations = None
        self.enable_express = None
        self.enable_partitioning = None
        self.forward_dead_lettered_messages_to = None
        self.forward_to = None
        self.lock_duration_in_seconds = None
        self.max_delivery_count = None
        self.max_size_in_mb = None
        self.requires_duplicate_detection = None
        self.status = None
        self.requires_session = None
        self.max_message_size_in_kb = None
        self.results = dict(changed=False, id=None)
        super(AzureRMServiceBusQueue, self).__init__(self.module_arg_spec, supports_tags=False, supports_check_mode=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()):
            setattr(self, key, kwargs[key])
        changed = False
        original = self.get()
        if self.state == 'present':
            params = dict(dead_lettering_on_message_expiration=self.dead_lettering_on_message_expiration, enable_batched_operations=self.enable_batched_operations, enable_express=self.enable_express, enable_partitioning=self.enable_partitioning, forward_dead_lettered_messages_to=self.forward_dead_lettered_messages_to, forward_to=self.forward_to, max_delivery_count=self.max_delivery_count, max_message_size_in_kilobytes=self.max_message_size_in_kb, max_size_in_megabytes=self.max_size_in_mb, requires_session=self.requires_session, requires_duplicate_detection=self.requires_duplicate_detection)
            if self.status:
                params['status'] = self.servicebus_models.EntityStatus(str.capitalize(_snake_to_camel(self.status)))
            for k, v in duration_spec_map.items():
                seconds = getattr(self, v)
                if seconds:
                    params[k] = timedelta(seconds=seconds)
            instance = self.servicebus_models.SBQueue(**params)
            result = original
            if not original:
                changed = True
                result = instance
            else:
                result = original
                attribute_map = set(self.servicebus_models.SBQueue._attribute_map.keys()) - set(self.servicebus_models.SBQueue._validation.keys())
                for attribute in attribute_map:
                    value = getattr(instance, attribute)
                    if value and value != getattr(original, attribute):
                        changed = True
            if changed and (not self.check_mode):
                result = self.create_or_update(instance)
            self.results = self.to_dict(result)
        elif original:
            changed = True
            if not self.check_mode:
                self.delete()
                self.results['deleted'] = True
        self.results['changed'] = changed
        return self.results

    def create_or_update(self, param):
        try:
            client = self._get_client()
            return client.create_or_update(self.resource_group, self.namespace, self.name, param)
        except Exception as exc:
            self.fail('Error creating or updating queue {0} - {1}'.format(self.name, str(exc.inner_exception) or str(exc)))

    def delete(self):
        try:
            client = self._get_client()
            client.delete(self.resource_group, self.namespace, self.name)
            return True
        except Exception as exc:
            self.fail('Error deleting queue {0} - {1}'.format(self.name, str(exc)))

    def _get_client(self):
        return self.servicebus_client.queues

    def get(self):
        try:
            client = self._get_client()
            return client.get(self.resource_group, self.namespace, self.name)
        except Exception:
            return None

    def to_dict(self, instance):
        result = dict()
        attribute_map = self.servicebus_models.SBQueue._attribute_map
        for attribute in attribute_map.keys():
            value = getattr(instance, attribute)
            if not value:
                continue
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
            elif attribute == 'max_size_in_kilobytes':
                result['max_size_in_kb'] = value
            else:
                result[attribute] = value
        return result