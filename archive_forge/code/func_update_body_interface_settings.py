from __future__ import absolute_import, division, print_function
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
from ansible.module_utils import six
def update_body_interface_settings(self):
    """Update network interface settings."""
    change_required = False
    if self.config_method == 'dhcp':
        if self.interface_info['config_method'] != 'configDhcp':
            if self.interface_info['address'] in self.url:
                self.use_alternate_address = True
            change_required = True
        self.body.update({'ipv4AddressConfigMethod': 'configDhcp'})
    else:
        self.body.update({'ipv4AddressConfigMethod': 'configStatic', 'ipv4Address': self.address, 'ipv4SubnetMask': self.subnet_mask})
        if self.interface_info['config_method'] != 'configStatic':
            change_required = True
        if self.address and self.interface_info['address'] != self.address:
            if self.interface_info['address'] in self.url:
                self.use_alternate_address = True
            change_required = True
        if self.subnet_mask and self.interface_info['subnet_mask'] != self.subnet_mask:
            change_required = True
        if self.gateway and self.interface_info['gateway'] != self.gateway:
            self.body.update({'ipv4GatewayAddress': self.gateway})
            change_required = True
    return change_required