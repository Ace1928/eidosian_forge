from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
class AzureRMIoTDeviceFacts(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(name=dict(type='str', aliases=['device_id']), module_id=dict(type='str'), query=dict(type='str'), hub=dict(type='str', required=True), hub_policy_name=dict(type='str', required=True), hub_policy_key=dict(type='str', no_log=True, required=True), top=dict(type='int'))
        self.results = dict(changed=False, iot_devices=[])
        self.name = None
        self.module_id = None
        self.hub = None
        self.hub_policy_name = None
        self.hub_policy_key = None
        self.top = None
        self.query = None
        self.mgmt_client = None
        self._base_url = None
        super(AzureRMIoTDeviceFacts, self).__init__(self.module_arg_spec, supports_check_mode=True)

    def exec_module(self, **kwargs):
        for key in self.module_arg_spec.keys():
            setattr(self, key, kwargs[key])
        self._base_url = '{0}.azure-devices.net'.format(self.hub)
        connect_str = 'HostName={0};SharedAccessKeyName={1};SharedAccessKey={2}'.format(self._base_url, self.hub_policy_name, self.hub_policy_key)
        self.mgmt_client = IoTHubRegistryManager.from_connection_string(connect_str)
        response = []
        if self.module_id:
            response = [self.get_device_module()]
        elif self.name:
            response = [self.get_device()]
        elif self.query is not None:
            response = self.hub_query()
        else:
            response = self.list_devices()
        self.results['iot_devices'] = response
        return self.results

    def get_device(self):
        try:
            response = self.mgmt_client.get_device(self.name)
            response = self.format_item(response)
            return response
        except Exception as exc:
            self.fail('Error when getting IoT Hub device {0}: {1}'.format(self.name, exc))

    def get_device_module(self):
        try:
            response = self.mgmt_client.get_module(self.name, self.module_id)
            return self.format_module(response)
        except Exception as exc:
            self.fail('Error when getting IoT Hub device {0}: {1}'.format(self.name, exc))

    def list_device_modules(self):
        try:
            response = self.mgmt_client.get_modules(self.name)
            return [self.format_module(item) for item in response]
        except Exception as exc:
            self.fail('Error when getting IoT Hub device {0}: {1}'.format(self.name, exc))

    def list_devices(self):
        try:
            response = None
            response = self.mgmt_client.get_devices(max_number_of_devices=1000)
            response = [self.format_item(item) for item in response]
            if self.top is not None:
                return response[self.top - 1]
            else:
                return response
        except Exception as exc:
            if hasattr(exc, 'message'):
                pass
            else:
                self.fail('Error when listing IoT Hub devices in {0}: {1}'.format(self.hub, exc))

    def hub_query(self):
        try:
            response = None
            response = self.mgmt_client.query_iot_hub(dict(query=self.query))
            return [self.format_twin(item) for item in response.items]
        except Exception as exc:
            if hasattr(exc, 'message'):
                pass
            else:
                self.fail('Error when listing IoT Hub devices in {0}: {1}'.format(self.hub, exc))

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

    def format_item(self, item):
        if not item:
            return None
        format_item = dict(authentication=dict(), capabilities=dict(), cloudToDeviceMessageCount=item.cloud_to_device_message_count, connectionState=item.connection_state, connectionStateUpdatedTime=item.connection_state_updated_time, deviceId=item.device_id, etag=item.etag, generationId=item.generation_id, lastActivityTime=item.last_activity_time, status=item.status, statusReason=item.status_reason)
        if hasattr(item, 'status_updated_time'):
            format_item['statusUpdatedTime'] = item.status_updated_time
        if hasattr(item, 'modules'):
            format_item['modules'] = item.modules
        if item.authentication:
            format_item['authentication']['symmetricKey'] = dict()
            format_item['authentication']['symmetricKey']['primaryKey'] = item.authentication.symmetric_key.primary_key
            format_item['authentication']['symmetricKey']['secondaryKey'] = item.authentication.symmetric_key.secondary_key
            format_item['authentication']['type'] = item.authentication.type
            format_item['authentication']['x509Thumbprint'] = dict()
            format_item['authentication']['x509Thumbprint']['primaryThumbprint'] = item.authentication.x509_thumbprint.primary_thumbprint
            format_item['authentication']['x509Thumbprint']['secondaryThumbprint'] = item.authentication.x509_thumbprint.secondary_thumbprint
        if item.capabilities:
            format_item['capabilities']['iotEdge'] = item.capabilities.iot_edge
        return format_item

    def format_twin(self, item):
        if not item:
            return None
        format_twin = dict(device_id=item.device_id, module_id=item.module_id, tags=item.tags, properties=dict(), etag=item.etag, version=item.version, device_etag=item.device_etag, status=item.status, cloud_to_device_message_count=item.cloud_to_device_message_count, authentication_type=item.authentication_type)
        if item.properties is not None:
            format_twin['properties']['desired'] = item.properties.desired
            format_twin['properties']['reported'] = item.properties.reported
        return format_twin