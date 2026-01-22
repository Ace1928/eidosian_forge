from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.facts.network.base import NetworkCollector
from ansible.module_utils.facts.network.generic_bsd import GenericBsdIfconfigNetwork
class AIXNetwork(GenericBsdIfconfigNetwork):
    """
    This is the AIX Network Class.
    It uses the GenericBsdIfconfigNetwork unchanged.
    """
    platform = 'AIX'

    def get_default_interfaces(self, route_path):
        interface = dict(v4={}, v6={})
        netstat_path = self.module.get_bin_path('netstat')
        if netstat_path:
            rc, out, err = self.module.run_command([netstat_path, '-nr'])
            lines = out.splitlines()
            for line in lines:
                words = line.split()
                if len(words) > 1 and words[0] == 'default':
                    if '.' in words[1]:
                        interface['v4']['gateway'] = words[1]
                        interface['v4']['interface'] = words[5]
                    elif ':' in words[1]:
                        interface['v6']['gateway'] = words[1]
                        interface['v6']['interface'] = words[5]
        return (interface['v4'], interface['v6'])

    def get_interfaces_info(self, ifconfig_path, ifconfig_options='-a'):
        interfaces = {}
        current_if = {}
        ips = dict(all_ipv4_addresses=[], all_ipv6_addresses=[])
        uname_rc = None
        uname_out = None
        uname_err = None
        uname_path = self.module.get_bin_path('uname')
        if uname_path:
            uname_rc, uname_out, uname_err = self.module.run_command([uname_path, '-W'])
        rc, out, err = self.module.run_command([ifconfig_path, ifconfig_options])
        for line in out.splitlines():
            if line:
                words = line.split()
                if re.match('^\\w*\\d*:', line):
                    current_if = self.parse_interface_line(words)
                    interfaces[current_if['device']] = current_if
                elif words[0].startswith('options='):
                    self.parse_options_line(words, current_if, ips)
                elif words[0] == 'nd6':
                    self.parse_nd6_line(words, current_if, ips)
                elif words[0] == 'ether':
                    self.parse_ether_line(words, current_if, ips)
                elif words[0] == 'media:':
                    self.parse_media_line(words, current_if, ips)
                elif words[0] == 'status:':
                    self.parse_status_line(words, current_if, ips)
                elif words[0] == 'lladdr':
                    self.parse_lladdr_line(words, current_if, ips)
                elif words[0] == 'inet':
                    self.parse_inet_line(words, current_if, ips)
                elif words[0] == 'inet6':
                    self.parse_inet6_line(words, current_if, ips)
                else:
                    self.parse_unknown_line(words, current_if, ips)
            if not uname_rc and uname_out.split()[0] == '0':
                if current_if['macaddress'] == 'unknown' and re.match('^en', current_if['device']):
                    entstat_path = self.module.get_bin_path('entstat')
                    if entstat_path:
                        rc, out, err = self.module.run_command([entstat_path, current_if['device']])
                        if rc != 0:
                            break
                        for line in out.splitlines():
                            if not line:
                                pass
                            buff = re.match('^Hardware Address: (.*)', line)
                            if buff:
                                current_if['macaddress'] = buff.group(1)
                            buff = re.match('^Device Type:', line)
                            if buff and re.match('.*Ethernet', line):
                                current_if['type'] = 'ether'
                if 'mtu' not in current_if:
                    lsattr_path = self.module.get_bin_path('lsattr')
                    if lsattr_path:
                        rc, out, err = self.module.run_command([lsattr_path, '-El', current_if['device']])
                        if rc != 0:
                            break
                        for line in out.splitlines():
                            if line:
                                words = line.split()
                                if words[0] == 'mtu':
                                    current_if['mtu'] = words[1]
        return (interfaces, ips)

    def parse_interface_line(self, words):
        device = words[0][0:-1]
        current_if = {'device': device, 'ipv4': [], 'ipv6': [], 'type': 'unknown'}
        current_if['flags'] = self.get_options(words[1])
        current_if['macaddress'] = 'unknown'
        return current_if