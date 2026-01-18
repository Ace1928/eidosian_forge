from __future__ import absolute_import, division, print_function
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
from ansible.module_utils import six
def update_body_enable_interface_setting(self):
    """Enable or disable the IPv4 network interface."""
    change_required = False
    if not self.enable_interface and (not self.interface_info['ipv6_enabled']):
        self.module.fail_json(msg='Either IPv4 or IPv6 must be enabled. Array [%s].' % self.ssid)
    if self.enable_interface != self.interface_info['enabled']:
        change_required = True
    self.body.update({'ipv4Enabled': self.enable_interface})
    return change_required