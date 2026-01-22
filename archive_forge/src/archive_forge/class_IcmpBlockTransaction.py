from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.parsing.convert_bool import boolean
from ansible_collections.ansible.posix.plugins.module_utils.firewalld import FirewallTransaction, fw_offline
class IcmpBlockTransaction(FirewallTransaction):
    """
    IcmpBlockTransaction
    """

    def __init__(self, module, action_args=None, zone=None, desired_state=None, permanent=False, immediate=False):
        super(IcmpBlockTransaction, self).__init__(module, action_args=action_args, desired_state=desired_state, zone=zone, permanent=permanent, immediate=immediate)

    def get_enabled_immediate(self, icmp_block, timeout):
        return icmp_block in self.fw.getIcmpBlocks(self.zone)

    def get_enabled_permanent(self, icmp_block, timeout):
        fw_zone, fw_settings = self.get_fw_zone_settings()
        return icmp_block in fw_settings.getIcmpBlocks()

    def set_enabled_immediate(self, icmp_block, timeout):
        self.fw.addIcmpBlock(self.zone, icmp_block, timeout)

    def set_enabled_permanent(self, icmp_block, timeout):
        fw_zone, fw_settings = self.get_fw_zone_settings()
        fw_settings.addIcmpBlock(icmp_block)
        self.update_fw_settings(fw_zone, fw_settings)

    def set_disabled_immediate(self, icmp_block, timeout):
        self.fw.removeIcmpBlock(self.zone, icmp_block)

    def set_disabled_permanent(self, icmp_block, timeout):
        fw_zone, fw_settings = self.get_fw_zone_settings()
        fw_settings.removeIcmpBlock(icmp_block)
        self.update_fw_settings(fw_zone, fw_settings)