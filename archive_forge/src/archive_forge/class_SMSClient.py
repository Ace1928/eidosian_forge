from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_sms import SMS
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec
class SMSClient(SMS):

    def __init__(self, module):
        super(SMSClient, self).__init__(module)

    def get_vasa_provider_info(self):
        self.get_sms_connection()
        results = dict(changed=False, vasa_providers=[])
        storage_manager = self.sms_si.QueryStorageManager()
        storage_providers = storage_manager.QueryProvider()
        for provider in storage_providers:
            provider_info = provider.QueryProviderInfo()
            temp_provider_info = {'name': provider_info.name, 'uid': provider_info.uid, 'description': provider_info.description, 'version': provider_info.version, 'certificate_status': provider_info.certificateStatus, 'url': provider_info.url, 'status': provider_info.status, 'related_storage_array': []}
            for a in provider_info.relatedStorageArray:
                temp_storage_array = {'active': str(a.active), 'array_id': a.arrayId, 'manageable': str(a.manageable), 'priority': str(a.priority)}
                temp_provider_info['related_storage_array'].append(temp_storage_array)
            results['vasa_providers'].append(temp_provider_info)
        self.module.exit_json(**results)