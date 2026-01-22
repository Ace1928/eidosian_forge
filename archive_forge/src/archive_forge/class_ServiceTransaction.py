from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.parsing.convert_bool import boolean
from ansible_collections.ansible.posix.plugins.module_utils.firewalld import FirewallTransaction, fw_offline
class ServiceTransaction(FirewallTransaction):
    """
    ServiceTransaction
    """

    def __init__(self, module, action_args=None, zone=None, desired_state=None, permanent=False, immediate=False):
        super(ServiceTransaction, self).__init__(module, action_args=action_args, desired_state=desired_state, zone=zone, permanent=permanent, immediate=immediate)

    def get_enabled_immediate(self, service, timeout):
        if service in self.fw.getServices(self.zone):
            return True
        else:
            return False

    def get_enabled_permanent(self, service, timeout):
        fw_zone, fw_settings = self.get_fw_zone_settings()
        if service in fw_settings.getServices():
            return True
        else:
            return False

    def set_enabled_immediate(self, service, timeout):
        self.fw.addService(self.zone, service, timeout)

    def set_enabled_permanent(self, service, timeout):
        fw_zone, fw_settings = self.get_fw_zone_settings()
        fw_settings.addService(service)
        self.update_fw_settings(fw_zone, fw_settings)

    def set_disabled_immediate(self, service, timeout):
        self.fw.removeService(self.zone, service)

    def set_disabled_permanent(self, service, timeout):
        fw_zone, fw_settings = self.get_fw_zone_settings()
        fw_settings.removeService(service)
        self.update_fw_settings(fw_zone, fw_settings)