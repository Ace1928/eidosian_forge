from __future__ import absolute_import, division, print_function
import re
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMIoTDeviceModule(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(name=dict(type='str', required=True), hub_policy_name=dict(type='str', required=True), hub_policy_key=dict(type='str', no_log=True, required=True), hub=dict(type='str', required=True), device=dict(type='str', required=True), state=dict(type='str', default='present', choices=['present', 'absent']), twin_tags=dict(type='dict'), desired=dict(type='dict'), auth_method=dict(type='str', choices=['self_signed', 'sas', 'certificate_authority'], default='sas'), primary_key=dict(type='str', no_log=True, aliases=['primary_thumbprint']), secondary_key=dict(type='str', no_log=True, aliases=['secondary_thumbprint']))
        self.results = dict(changed=False, id=None)
        self.name = None
        self.hub = None
        self.device = None
        self.hub_policy_key = None
        self.hub_policy_name = None
        self.state = None
        self.twin_tags = None
        self.desired = None
        self.auth_method = None
        self.primary_key = None
        self.secondary_key = None
        self.managed_by = None
        self.etag = None
        self._base_url = None
        self.mgmt_client = None
        super(AzureRMIoTDeviceModule, self).__init__(self.module_arg_spec, supports_check_mode=True)

    def exec_module(self, **kwargs):
        for key in self.module_arg_spec.keys():
            setattr(self, key, kwargs[key])
        self._base_url = '{0}.azure-devices.net'.format(self.hub)
        connect_str = 'HostName={0};SharedAccessKeyName={1};SharedAccessKey={2}'.format(self._base_url, self.hub_policy_name, self.hub_policy_key)
        self.mgmt_client = IoTHubRegistryManager.from_connection_string(connect_str)
        changed = False
        module = self.get_module()
        if self.state == 'present':
            if not module:
                changed = True
                if not self.check_mode:
                    module = self.create_module()
            else:
                self.etag = module.get('etag')
            twin = self.get_twin()
            if not twin.get('tags'):
                twin['tags'] = dict()
            twin_change = False
            if self.twin_tags and (not self.is_equal(self.twin_tags, twin['tags'])):
                twin_change = True
            if self.desired and (not self.is_equal(self.desired, twin['properties']['desired'])):
                self.module.warn('desired')
                twin_change = True
            if twin_change and (not self.check_mode):
                twin_dict = dict()
                twin_dict['tags'] = self.twin_tags
                twin_dict['properties'] = dict()
                twin_dict['properties']['desired'] = self.desired
                twin = self.update_twin(twin_dict)
            changed = changed or twin_change
            module['tags'] = twin.get('tags') or dict()
            module['properties'] = twin.get('properties') or dict()
        elif module and (not self.check_mode):
            self.delete_module()
            changed = True
        self.results = module or dict()
        self.results['changed'] = changed
        return self.results

    def is_equal(self, updated, original):
        changed = False
        if not isinstance(updated, dict):
            self.fail('The Property or Tag should be a dict')
        for key in updated.keys():
            if re.search('[.|$|#|\\s]', key):
                self.fail("Property or Tag name has invalid characters: '.', '$', '#' or ' '. Got '{0}'".format(key))
            original_value = original.get(key)
            updated_value = updated[key]
            if isinstance(updated_value, dict):
                if not isinstance(original_value, dict):
                    changed = True
                    original[key] = updated_value
                elif not self.is_equal(updated_value, original_value):
                    changed = True
            elif original_value != updated_value:
                changed = True
                original[key] = updated_value
        return not changed

    def update_module(self):
        response = None
        try:
            if self.auth_method == 'sas':
                response = self.mgmt_client.update_module_with_sas(self.device, self.name, self.managed_by, self.etag, self.primary_key, self.secondary_key)
            elif self.auth_method == 'self_signed':
                response = self.mgmt_client.update_module_with_certificate_authority(self.device, self.name, self.managed_by, self.etag)
            elif self.auth_method == 'certificate_authority':
                response = self.mgmt_client.update_module_with_x509(self.device, self.name, self.managed_by, self.etag, self.primary_thumbprint, self.secondary_thumbprint)
            return self.format_module(response)
        except Exception as exc:
            if exc.status_code in [403] and self.edge_enabled:
                self.fail('Edge device is not supported in IoT Hub with Basic tier.')
            else:
                self.fail('Error when creating or updating IoT Hub device {0}: {1}'.format(self.name, exc.message or str(exc)))

    def create_module(self):
        response = None
        try:
            if self.auth_method == 'sas':
                response = self.mgmt_client.create_module_with_sas(self.device, self.name, self.managed_by, self.primary_key, self.secondary_key)
            elif self.auth_method == 'self_signed':
                response = self.mgmt_client.create_module_with_certificate_authority(self.device, self.name, self.managed_by)
            elif self.auth_method == 'certificate_authority':
                response = self.mgmt_client.create_module_with_x509(self.device_id, self.name, self.managed_by, self.primary_thumbprint, self.secondary_thumbprint)
            return self.format_module(response)
        except Exception as exc:
            self.fail('Error when creating or updating IoT Hub device {0}: {1}'.format(self.name, exc.message or str(exc)))

    def delete_module(self, etag):
        try:
            response = self.mgmt_client.delete_module(self.device, self.name)
            return response
        except Exception as exc:
            self.fail('Error when deleting IoT Hub device {0}: {1}'.format(self.name, exc.message or str(exc)))

    def get_module(self):
        try:
            response = self.mgmt_client.get_module(self.device, self.name)
            return self.format_module(response)
        except Exception:
            return None

    def get_twin(self):
        try:
            response = self.mgmt_client.get_twin(self.device)
            return self.format_twin(response)
        except Exception as exc:
            self.fail('Error when getting IoT Hub device {0} module twin {1}: {2}'.format(self.device, self.name, exc.message or str(exc)))

    def update_twin(self, twin):
        try:
            response = self.mgmt_client.update_twin(self.device, twin)
            return self.format_twin(response)
        except Exception as exc:
            self.fail('Error when creating or updating IoT Hub device {0} module twin {1}: {2}'.format(self.device, self.name, exc.message or str(exc)))

    def format_module(self, item):
        if not item:
            return None
        format_item = dict(authentication=dict(), cloudToDeviceMessageCount=item.cloud_to_device_message_count, connectionState=item.connection_state, connectionStateUpdatedTime=item.connection_state_updated_time, deviceId=item.device_id, etag=item.etag, generationId=item.generation_id, lastActivityTime=item.last_activity_time, managedBy=item.managed_by, moduleId=item.module_id)
        if item.authentication:
            format_item['authentication']['symmetricKey'] = dict()
            format_item['authentication']['symmetricKey']['primaryKey'] = item.authentication.symmetric_key.primary_key
            format_item['authentication']['symmetricKey']['secondaryKey'] = item.authentication.symmetric_key.secondary_key
            format_item['authentication']['type'] = item.authentication.type
            format_item['authentication']['x509Thumbprint'] = dict()
            format_item['authentication']['x509Thumbprint']['primaryThumbprint'] = item.authentication.x509_thumbprint.primary_thumbprint
            format_item['authentication']['x509Thumbprint']['secondaryThumbprint'] = item.authentication.x509_thumbprint.secondary_thumbprint
        return format_item

    def format_twin(self, item):
        if not item:
            return None
        format_twin = dict(device_id=item.device_id, module_id=item.module_id, tags=item.tags, properties=dict(), etag=item.etag, version=item.version, device_etag=item.device_etag, status=item.status, cloud_to_device_message_count=item.cloud_to_device_message_count, authentication_type=item.authentication_type)
        if item.properties is not None:
            format_twin['properties']['desired'] = item.properties.desired
            format_twin['properties']['reported'] = item.properties.reported
        return format_twin