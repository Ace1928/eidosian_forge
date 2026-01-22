from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware_sms import (
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec
from ansible.module_utils._text import to_native
class VMwareVASA(SMS):

    def __init__(self, module):
        super(VMwareVASA, self).__init__(module)
        self.vasa_name = module.params['vasa_name']
        self.vasa_url = module.params['vasa_url']
        self.vasa_username = module.params['vasa_username']
        self.vasa_password = module.params['vasa_password']
        self.vasa_certificate = module.params['vasa_certificate']
        self.desired_state = module.params['state']
        self.storage_manager = None

    def process_state(self):
        """
        Manage internal states of VASA provider
        """
        vasa_states = {'absent': {'present': self.state_unregister_vasa, 'absent': self.state_exit_unchanged}, 'present': {'present': self.state_exit_unchanged, 'absent': self.state_register_vasa}}
        self.get_sms_connection()
        current_state = self.check_vasa_configuration()
        vasa_states[self.desired_state][current_state]()

    def state_register_vasa(self):
        """
        Register VASA provider with vcenter
        """
        changed, result = (True, None)
        vasa_provider_spec = sms.provider.VasaProviderSpec()
        vasa_provider_spec.name = self.vasa_name
        vasa_provider_spec.username = self.vasa_username
        vasa_provider_spec.password = self.vasa_password
        vasa_provider_spec.url = self.vasa_url
        vasa_provider_spec.certificate = self.vasa_certificate
        try:
            if not self.module.check_mode:
                task = self.storage_manager.RegisterProvider_Task(vasa_provider_spec)
                changed, result = wait_for_sms_task(task)
                if isinstance(result, sms.fault.CertificateNotTrusted):
                    vasa_provider_spec.certificate = result.certificate
                    task = self.storage_manager.RegisterProvider_Task(vasa_provider_spec)
                    changed, result = wait_for_sms_task(task)
                if isinstance(result, sms.provider.VasaProvider):
                    provider_info = result.QueryProviderInfo()
                    temp_provider_info = {'name': provider_info.name, 'uid': provider_info.uid, 'description': provider_info.description, 'version': provider_info.version, 'certificate_status': provider_info.certificateStatus, 'url': provider_info.url, 'status': provider_info.status, 'related_storage_array': []}
                    for a in provider_info.relatedStorageArray:
                        temp_storage_array = {'active': str(a.active), 'array_id': a.arrayId, 'manageable': str(a.manageable), 'priority': str(a.priority)}
                        temp_provider_info['related_storage_array'].append(temp_storage_array)
                    result = temp_provider_info
            self.module.exit_json(changed=changed, result=result)
        except TaskError as task_err:
            self.module.fail_json(msg='Failed to register VASA provider due to task exception %s' % to_native(task_err))
        except Exception as generic_exc:
            self.module.fail_json(msg='Failed to register VASA due to generic exception %s' % to_native(generic_exc))

    def state_unregister_vasa(self):
        """
        Unregister VASA provider
        """
        changed, result = (True, None)
        try:
            if not self.module.check_mode:
                uid = self.vasa_provider_info.uid
                task = self.storage_manager.UnregisterProvider_Task(uid)
                changed, result = wait_for_sms_task(task)
            self.module.exit_json(changed=changed, result=result)
        except Exception as generic_exc:
            self.module.fail_json(msg='Failed to unregister VASA due to generic exception %s' % to_native(generic_exc))

    def state_exit_unchanged(self):
        """
        Exit without any change
        """
        self.module.exit_json(changed=False)

    def check_vasa_configuration(self):
        """
        Check VASA configuration
        Returns: 'Present' if VASA provider exists, else 'absent'

        """
        self.vasa_provider_info = None
        self.storage_manager = self.sms_si.QueryStorageManager()
        storage_providers = self.storage_manager.QueryProvider()
        try:
            for provider in storage_providers:
                provider_info = provider.QueryProviderInfo()
                if provider_info.name == self.vasa_name:
                    if provider_info.url != self.vasa_url:
                        raise Exception("VASA provider '%s' URL '%s' is inconsistent  with task parameter '%s'" % (self.vasa_name, provider_info.url, self.vasa_url))
                    self.vasa_provider_info = provider_info
                    break
            if self.vasa_provider_info is None:
                return 'absent'
            return 'present'
        except Exception as generic_exc:
            self.module.fail_json(msg='Failed to check configuration due to generic exception %s' % to_native(generic_exc))