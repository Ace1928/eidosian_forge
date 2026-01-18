from __future__ import absolute_import, division, print_function
from time import sleep
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
from ansible.module_utils import six
def update_body_dns_server_settings(self):
    """Add DNS server information to the request body."""
    change_required = False
    if self.dns_config_method == 'dhcp':
        if self.interface_info['dns_config_method'] != 'dhcp':
            change_required = True
        self.body.update({'dnsAcquisitionDescriptor': {'dnsAcquisitionType': 'dhcp'}})
    elif self.dns_config_method == 'static':
        dns_servers = []
        if self.dns_address and self.dns_address_backup and (not self.interface_info['dns_servers'] or len(self.interface_info['dns_servers']) != 2) or (self.dns_address and (not self.dns_address_backup) and (not self.interface_info['dns_servers'] or len(self.interface_info['dns_servers']) != 1)):
            change_required = True
        if self.dns_address:
            if is_ipv4(self.dns_address):
                dns_servers.append({'addressType': 'ipv4', 'ipv4Address': self.dns_address})
                if not self.interface_info['dns_servers'] or len(self.interface_info['dns_servers']) < 1 or self.interface_info['dns_servers'][0]['addressType'] != 'ipv4' or (self.interface_info['dns_servers'][0]['ipv4Address'] != self.dns_address):
                    change_required = True
            elif is_ipv6(self.dns_address):
                dns_servers.append({'addressType': 'ipv6', 'ipv6Address': self.dns_address})
                if not self.interface_info['dns_servers'] or len(self.interface_info['dns_servers']) < 1 or self.interface_info['dns_servers'][0]['addressType'] != 'ipv6' or (self.interface_info['dns_servers'][0]['ipv6Address'].replace(':', '').lower() != self.dns_address.replace(':', '').lower()):
                    change_required = True
            else:
                self.module.fail_json(msg='Invalid IP address! DNS address must be either IPv4 or IPv6. Address [%s]. Array [%s].' % (self.dns_address, self.ssid))
        if self.dns_address_backup:
            if is_ipv4(self.dns_address_backup):
                dns_servers.append({'addressType': 'ipv4', 'ipv4Address': self.dns_address_backup})
                if not self.interface_info['dns_servers'] or len(self.interface_info['dns_servers']) < 2 or self.interface_info['dns_servers'][1]['addressType'] != 'ipv4' or (self.interface_info['dns_servers'][1]['ipv4Address'] != self.dns_address_backup):
                    change_required = True
            elif is_ipv6(self.dns_address_backup):
                dns_servers.append({'addressType': 'ipv6', 'ipv6Address': self.dns_address_backup})
                if not self.interface_info['dns_servers'] or len(self.interface_info['dns_servers']) < 2 or self.interface_info['dns_servers'][1]['addressType'] != 'ipv6' or (self.interface_info['dns_servers'][1]['ipv6Address'].replace(':', '').lower() != self.dns_address_backup.replace(':', '').lower()):
                    change_required = True
            else:
                self.module.fail_json(msg='Invalid IP address! DNS address must be either IPv4 or IPv6. Address [%s]. Array [%s].' % (self.dns_address, self.ssid))
        self.body.update({'dnsAcquisitionDescriptor': {'dnsAcquisitionType': 'stat', 'dnsServers': dns_servers}})
    return change_required