from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.parsing.convert_bool import boolean
from ansible_collections.ansible.posix.plugins.module_utils.firewalld import FirewallTransaction, fw_offline
class ZoneTargetTransaction(FirewallTransaction):
    """
    ZoneTargetTransaction
    """

    def __init__(self, module, action_args=None, zone=None, desired_state=None, permanent=True, immediate=False, enabled_values=None, disabled_values=None):
        super(ZoneTargetTransaction, self).__init__(module, action_args=action_args, desired_state=desired_state, zone=zone, permanent=permanent, immediate=immediate, enabled_values=enabled_values or ['present', 'enabled'], disabled_values=disabled_values or ['absent', 'disabled'])
        self.enabled_msg = 'Set zone %s target to %s' % (self.zone, action_args[0])
        self.disabled_msg = 'Reset zone %s target to default' % self.zone
        self.tx_not_permanent_error_msg = "Zone operations must be permanent. Make sure you didn't set the 'permanent' flag to 'false' or the 'immediate' flag to 'true'."

    def get_enabled_immediate(self, target):
        self.module.fail_json(msg=self.tx_not_permanent_error_msg)

    def get_enabled_permanent(self, target):
        fw_zone, fw_settings = self.get_fw_zone_settings()
        current_target = fw_settings.getTarget()
        return current_target == target

    def set_enabled_immediate(self, target):
        self.module.fail_json(msg=self.tx_not_permanent_error_msg)

    def set_enabled_permanent(self, target):
        fw_zone, fw_settings = self.get_fw_zone_settings()
        fw_settings.setTarget(target)
        self.update_fw_settings(fw_zone, fw_settings)

    def set_disabled_immediate(self, target):
        self.module.fail_json(msg=self.tx_not_permanent_error_msg)

    def set_disabled_permanent(self, target):
        fw_zone, fw_settings = self.get_fw_zone_settings()
        fw_settings.setTarget('default')
        self.update_fw_settings(fw_zone, fw_settings)