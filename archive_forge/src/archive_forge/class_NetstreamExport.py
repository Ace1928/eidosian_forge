from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import exec_command, load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
class NetstreamExport(object):
    """Manage NetStream export"""

    def __init__(self, argument_spec):
        self.spec = argument_spec
        self.module = None
        self.__init_module__()
        self.type = self.module.params['type']
        self.source_ip = self.module.params['source_ip']
        self.host_ip = self.module.params['host_ip']
        self.host_port = self.module.params['host_port']
        self.host_vpn = self.module.params['host_vpn']
        self.version = self.module.params['version']
        self.as_option = self.module.params['as_option']
        self.bgp_netxhop = self.module.params['bgp_nexthop']
        self.state = self.module.params['state']
        self.commands = list()
        self.config = None
        self.exist_conf = dict()
        self.changed = False
        self.updates_cmd = list()
        self.results = dict()
        self.proposed = dict()
        self.existing = dict()
        self.end_state = dict()

    def __init_module__(self):
        """init module"""
        self.module = AnsibleModule(argument_spec=self.spec, supports_check_mode=True)

    def cli_load_config(self, commands):
        """load config by cli"""
        if not self.module.check_mode:
            load_config(self.module, commands)

    def get_netstream_config(self):
        """get current netstream configuration"""
        cmd = 'display current-configuration | include ^netstream export'
        rc, out, err = exec_command(self.module, cmd)
        if rc != 0:
            self.module.fail_json(msg=err)
        config = str(out).strip()
        return config

    def get_existing(self):
        """get existing config"""
        self.existing = dict(type=self.type, source_ip=self.exist_conf['source_ip'], host_ip=self.exist_conf['host_ip'], host_port=self.exist_conf['host_port'], host_vpn=self.exist_conf['host_vpn'], version=self.exist_conf['version'], as_option=self.exist_conf['as_option'], bgp_nexthop=self.exist_conf['bgp_netxhop'])

    def get_proposed(self):
        """get proposed config"""
        self.proposed = dict(type=self.type, source_ip=self.source_ip, host_ip=self.host_ip, host_port=self.host_port, host_vpn=self.host_vpn, version=self.version, as_option=self.as_option, bgp_nexthop=self.bgp_netxhop, state=self.state)

    def get_end_state(self):
        """get end config"""
        self.get_config_data()
        self.end_state = dict(type=self.type, source_ip=self.exist_conf['source_ip'], host_ip=self.exist_conf['host_ip'], host_port=self.exist_conf['host_port'], host_vpn=self.exist_conf['host_vpn'], version=self.exist_conf['version'], as_option=self.exist_conf['as_option'], bgp_nexthop=self.exist_conf['bgp_netxhop'])

    def show_result(self):
        """show result"""
        self.results['changed'] = self.changed
        self.results['proposed'] = self.proposed
        self.results['existing'] = self.existing
        self.results['end_state'] = self.end_state
        if self.changed:
            self.results['updates'] = self.updates_cmd
        else:
            self.results['updates'] = list()
        self.module.exit_json(**self.results)

    def cli_add_command(self, command, undo=False):
        """add command to self.update_cmd and self.commands"""
        if undo and command.lower() not in ['quit', 'return']:
            cmd = 'undo ' + command
        else:
            cmd = command
        self.commands.append(cmd)
        if command.lower() not in ['quit', 'return']:
            if cmd not in self.updates_cmd:
                self.updates_cmd.append(cmd)

    def config_nets_export_src_addr(self):
        """Configures the source address for the exported packets"""
        if is_ipv4_addr(self.source_ip):
            if self.type == 'ip':
                cmd = 'netstream export ip source %s' % self.source_ip
            else:
                cmd = 'netstream export vxlan inner-ip source %s' % self.source_ip
        elif self.type == 'ip':
            cmd = 'netstream export ip source ipv6 %s' % self.source_ip
        else:
            cmd = 'netstream export vxlan inner-ip source ipv6 %s' % self.source_ip
        if is_config_exist(self.config, cmd):
            self.exist_conf['source_ip'] = self.source_ip
            if self.state == 'present':
                return
            else:
                undo = True
        elif self.state == 'absent':
            return
        else:
            undo = False
        self.cli_add_command(cmd, undo)

    def config_nets_export_host_addr(self):
        """Configures the destination IP address and destination UDP port number"""
        if is_ipv4_addr(self.host_ip):
            if self.type == 'ip':
                cmd = 'netstream export ip host %s %s' % (self.host_ip, self.host_port)
            else:
                cmd = 'netstream export vxlan inner-ip host %s %s' % (self.host_ip, self.host_port)
        elif self.type == 'ip':
            cmd = 'netstream export ip host ipv6 %s %s' % (self.host_ip, self.host_port)
        else:
            cmd = 'netstream export vxlan inner-ip host ipv6 %s %s' % (self.host_ip, self.host_port)
        if self.host_vpn:
            cmd += ' vpn-instance %s' % self.host_vpn
        if is_config_exist(self.config, cmd):
            self.exist_conf['host_ip'] = self.host_ip
            self.exist_conf['host_port'] = self.host_port
            if self.host_vpn:
                self.exist_conf['host_vpn'] = self.host_vpn
            if self.state == 'present':
                return
            else:
                undo = True
        elif self.state == 'absent':
            return
        else:
            undo = False
        self.cli_add_command(cmd, undo)

    def config_nets_export_vxlan_ver(self):
        """Configures the version for the exported packets carrying VXLAN flexible flow statistics"""
        cmd = 'netstream export vxlan inner-ip version 9'
        if is_config_exist(self.config, cmd):
            self.exist_conf['version'] = self.version
            if self.state == 'present':
                return
            else:
                undo = True
        elif self.state == 'absent':
            return
        else:
            undo = False
        self.cli_add_command(cmd, undo)

    def config_nets_export_ip_ver(self):
        """Configures the version number of the exported packets carrying IPv4 flow statistics"""
        cmd = 'netstream export ip version %s' % self.version
        if self.version == '5':
            if self.as_option == 'origin':
                cmd += ' origin-as'
            elif self.as_option == 'peer':
                cmd += ' peer-as'
        else:
            if self.as_option == 'origin':
                cmd += ' origin-as'
            elif self.as_option == 'peer':
                cmd += ' peer-as'
            if self.bgp_netxhop == 'enable':
                cmd += ' bgp-nexthop'
        if cmd == 'netstream export ip version 5':
            cmd_tmp = 'netstream export ip version'
            if cmd_tmp in self.config:
                if self.state == 'present':
                    self.cli_add_command(cmd, False)
            else:
                self.exist_conf['version'] = self.version
            return
        if is_config_exist(self.config, cmd):
            self.exist_conf['version'] = self.version
            self.exist_conf['as_option'] = self.as_option
            self.exist_conf['bgp_netxhop'] = self.bgp_netxhop
            if self.state == 'present':
                return
            else:
                undo = True
        elif self.state == 'absent':
            return
        else:
            undo = False
        self.cli_add_command(cmd, undo)

    def config_netstream_export(self):
        """configure netstream export"""
        if self.commands:
            self.cli_load_config(self.commands)
            self.changed = True

    def check_params(self):
        """Check all input params"""
        if not self.type:
            self.module.fail_json(msg='Error: The value of type cannot be empty.')
        if self.host_port:
            if not self.host_port.isdigit():
                self.module.fail_json(msg='Error: Host port is invalid.')
            if int(self.host_port) < 1 or int(self.host_port) > 65535:
                self.module.fail_json(msg='Error: Host port is not in the range from 1 to 65535.')
        if self.host_vpn:
            if self.host_vpn == '_public_':
                self.module.fail_json(msg='Error: The host vpn name _public_ is reserved.')
            if len(self.host_vpn) < 1 or len(self.host_vpn) > 31:
                self.module.fail_json(msg='Error: The host vpn name length is not in the range from 1 to 31.')
        if self.type == 'vxlan' and self.version == '5':
            self.module.fail_json(msg='Error: When type is vxlan, version must be 9.')
        if self.type == 'ip' and self.version == '5' and (self.bgp_netxhop == 'enable'):
            self.module.fail_json(msg='Error: When type=ip and version=5, bgp_netxhop is not supported.')
        if self.host_ip and (not self.host_port) or (self.host_port and (not self.host_ip)):
            self.module.fail_json(msg='Error: host_ip and host_port must both exist or not exist.')

    def get_config_data(self):
        """get configuration commands and current configuration"""
        self.exist_conf['type'] = self.type
        self.exist_conf['source_ip'] = None
        self.exist_conf['host_ip'] = None
        self.exist_conf['host_port'] = None
        self.exist_conf['host_vpn'] = None
        self.exist_conf['version'] = None
        self.exist_conf['as_option'] = None
        self.exist_conf['bgp_netxhop'] = 'disable'
        self.config = self.get_netstream_config()
        if self.type and self.source_ip:
            self.config_nets_export_src_addr()
        if self.type and self.host_ip and self.host_port:
            self.config_nets_export_host_addr()
        if self.type == 'vxlan' and self.version == '9':
            self.config_nets_export_vxlan_ver()
        if self.type == 'ip' and self.version:
            self.config_nets_export_ip_ver()

    def work(self):
        """execute task"""
        self.check_params()
        self.get_proposed()
        self.get_config_data()
        self.get_existing()
        self.config_netstream_export()
        self.get_end_state()
        self.show_result()