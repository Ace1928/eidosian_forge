from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.parsing.convert_bool import boolean
from ansible_collections.ansible.posix.plugins.module_utils.firewalld import FirewallTransaction, fw_offline
class ProtocolTransaction(FirewallTransaction):
    """
    ProtocolTransaction
    """

    def __init__(self, module, action_args=None, zone=None, desired_state=None, permanent=False, immediate=False):
        super(ProtocolTransaction, self).__init__(module, action_args=action_args, desired_state=desired_state, zone=zone, permanent=permanent, immediate=immediate)

    def get_enabled_immediate(self, protocol, timeout):
        if protocol in self.fw.getProtocols(self.zone):
            return True
        else:
            return False

    def get_enabled_permanent(self, protocol, timeout):
        fw_zone, fw_settings = self.get_fw_zone_settings()
        if protocol in fw_settings.getProtocols():
            return True
        else:
            return False

    def set_enabled_immediate(self, protocol, timeout):
        self.fw.addProtocol(self.zone, protocol, timeout)

    def set_enabled_permanent(self, protocol, timeout):
        fw_zone, fw_settings = self.get_fw_zone_settings()
        fw_settings.addProtocol(protocol)
        self.update_fw_settings(fw_zone, fw_settings)

    def set_disabled_immediate(self, protocol, timeout):
        self.fw.removeProtocol(self.zone, protocol)

    def set_disabled_permanent(self, protocol, timeout):
        fw_zone, fw_settings = self.get_fw_zone_settings()
        fw_settings.removeProtocol(protocol)
        self.update_fw_settings(fw_zone, fw_settings)