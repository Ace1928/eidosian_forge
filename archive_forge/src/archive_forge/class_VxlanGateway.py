from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
from ansible.module_utils.connection import exec_command
class VxlanGateway(object):
    """
    Manages Gateway for the VXLAN Network.
    """

    def __init__(self, argument_spec):
        self.spec = argument_spec
        self.module = None
        self.init_module()
        self.dfs_id = self.module.params['dfs_id']
        self.dfs_source_ip = self.module.params['dfs_source_ip']
        self.dfs_source_vpn = self.module.params['dfs_source_vpn']
        self.dfs_udp_port = self.module.params['dfs_udp_port']
        self.dfs_all_active = self.module.params['dfs_all_active']
        self.dfs_peer_ip = self.module.params['dfs_peer_ip']
        self.dfs_peer_vpn = self.module.params['dfs_peer_vpn']
        self.vpn_instance = self.module.params['vpn_instance']
        self.vpn_vni = self.module.params['vpn_vni']
        self.vbdif_name = self.module.params['vbdif_name']
        self.vbdif_mac = self.module.params['vbdif_mac']
        self.vbdif_bind_vpn = self.module.params['vbdif_bind_vpn']
        self.arp_distribute_gateway = self.module.params['arp_distribute_gateway']
        self.arp_direct_route = self.module.params['arp_direct_route']
        self.state = self.module.params['state']
        self.host = self.module.params['host']
        self.username = self.module.params['username']
        self.port = self.module.params['port']
        self.config = ''
        self.changed = False
        self.updates_cmd = list()
        self.commands = list()
        self.results = dict()
        self.proposed = dict()
        self.existing = dict()
        self.end_state = dict()

    def init_module(self):
        """init module"""
        self.module = AnsibleModule(argument_spec=self.spec, supports_check_mode=True)

    def cli_load_config(self, commands):
        """load config by cli"""
        if not self.module.check_mode:
            load_config(self.module, commands)

    def get_config(self, flags=None):
        """Retrieves the current config from the device or cache
        """
        flags = [] if flags is None else flags
        cmd = 'display current-configuration '
        cmd += ' '.join(flags)
        cmd = cmd.strip()
        rc, out, err = exec_command(self.module, cmd)
        if rc != 0:
            self.module.fail_json(msg=err)
        cfg = str(out).strip()
        return cfg

    def get_current_config(self):
        """get current configuration"""
        flags = list()
        exp = ' | ignore-case section include ^#\\s+dfs-group'
        if self.vpn_instance:
            exp += '|^#\\s+ip vpn-instance %s' % self.vpn_instance
        if self.vbdif_name:
            exp += '|^#\\s+interface %s' % self.vbdif_name
        flags.append(exp)
        return self.get_config(flags)

    def cli_add_command(self, command, undo=False):
        """add command to self.update_cmd and self.commands"""
        if undo and command.lower() not in ['quit', 'return']:
            cmd = 'undo ' + command
        else:
            cmd = command
        self.commands.append(cmd)
        if command.lower() not in ['quit', 'return']:
            self.updates_cmd.append(cmd)

    def config_dfs_group(self):
        """manage Dynamic Fabric Service (DFS) group configuration"""
        if not self.dfs_id:
            return
        dfs_view = False
        view_cmd = 'dfs-group %s' % self.dfs_id
        exist = is_config_exist(self.config, view_cmd)
        if self.state == 'present' and (not exist):
            self.cli_add_command(view_cmd)
            dfs_view = True
        if self.state == 'absent' and exist:
            if not self.dfs_source_ip and (not self.dfs_udp_port) and (not self.dfs_all_active) and (not self.dfs_peer_ip):
                self.cli_add_command(view_cmd, undo=True)
                return
        if self.dfs_source_ip:
            cmd = 'source ip %s' % self.dfs_source_ip
            if self.dfs_source_vpn:
                cmd += ' vpn-instance %s' % self.dfs_source_vpn
            exist = is_config_exist(self.config, cmd)
            if self.state == 'present' and (not exist):
                if not dfs_view:
                    self.cli_add_command(view_cmd)
                    dfs_view = True
                self.cli_add_command(cmd)
            if self.state == 'absent' and exist:
                if not dfs_view:
                    self.cli_add_command(view_cmd)
                    dfs_view = True
                self.cli_add_command(cmd, undo=True)
        if self.dfs_udp_port:
            cmd = 'udp port %s' % self.dfs_udp_port
            exist = is_config_exist(self.config, cmd)
            if self.state == 'present' and (not exist):
                if not dfs_view:
                    self.cli_add_command(view_cmd)
                    dfs_view = True
                self.cli_add_command(cmd)
            elif self.state == 'absent' and exist:
                if not dfs_view:
                    self.cli_add_command(view_cmd)
                    dfs_view = True
                self.cli_add_command(cmd, undo=True)
        aa_cmd = 'active-active-gateway'
        aa_exist = is_config_exist(self.config, aa_cmd)
        aa_view = False
        if self.dfs_all_active == 'disable':
            if aa_exist:
                cmd = 'peer %s' % self.dfs_peer_ip
                if self.dfs_source_vpn:
                    cmd += ' vpn-instance %s' % self.dfs_peer_vpn
                exist = is_config_exist(self.config, cmd)
                if self.state == 'absent' and exist:
                    if not dfs_view:
                        self.cli_add_command(view_cmd)
                        dfs_view = True
                    self.cli_add_command(aa_cmd)
                    self.cli_add_command(cmd, undo=True)
                    self.cli_add_command('quit')
                if not dfs_view:
                    self.cli_add_command(view_cmd)
                    dfs_view = True
                self.cli_add_command(aa_cmd, undo=True)
        elif self.dfs_all_active == 'enable':
            if not aa_exist:
                if not dfs_view:
                    self.cli_add_command(view_cmd)
                    dfs_view = True
                self.cli_add_command(aa_cmd)
                aa_view = True
            if self.dfs_peer_ip:
                cmd = 'peer %s' % self.dfs_peer_ip
                if self.dfs_peer_vpn:
                    cmd += ' vpn-instance %s' % self.dfs_peer_vpn
                exist = is_config_exist(self.config, cmd)
                if self.state == 'present' and (not exist):
                    if not dfs_view:
                        self.cli_add_command(view_cmd)
                        dfs_view = True
                    if not aa_view:
                        self.cli_add_command(aa_cmd)
                    self.cli_add_command(cmd)
                    self.cli_add_command('quit')
                elif self.state == 'absent' and exist:
                    if not dfs_view:
                        self.cli_add_command(view_cmd)
                        dfs_view = True
                    if not aa_view:
                        self.cli_add_command(aa_cmd)
                    self.cli_add_command(cmd, undo=True)
                    self.cli_add_command('quit')
        elif aa_exist and self.dfs_peer_ip:
            cmd = 'peer %s' % self.dfs_peer_ip
            if self.dfs_peer_vpn:
                cmd += ' vpn-instance %s' % self.dfs_peer_vpn
            exist = is_config_exist(self.config, cmd)
            if self.state == 'present' and (not exist):
                if not dfs_view:
                    self.cli_add_command(view_cmd)
                    dfs_view = True
                self.cli_add_command(aa_cmd)
                self.cli_add_command(cmd)
                self.cli_add_command('quit')
            elif self.state == 'absent' and exist:
                if not dfs_view:
                    self.cli_add_command(view_cmd)
                    dfs_view = True
                self.cli_add_command(aa_cmd)
                self.cli_add_command(cmd, undo=True)
                self.cli_add_command('quit')
            else:
                pass
        elif not aa_exist and self.dfs_peer_ip and (self.state == 'present'):
            self.module.fail_json(msg='Error: All-active gateways is not enable.')
        else:
            pass
        if dfs_view:
            self.cli_add_command('quit')

    def config_ip_vpn(self):
        """configure command at the ip vpn view"""
        if not self.vpn_instance or not self.vpn_vni:
            return
        view_cmd = 'ip vpn-instance %s' % self.vpn_instance
        exist = is_config_exist(self.config, view_cmd)
        if not exist:
            self.module.fail_json(msg='Error: ip vpn instance %s is not exist.' % self.vpn_instance)
        cmd = 'vxlan vni %s' % self.vpn_vni
        exist = is_config_exist(self.config, cmd)
        if self.state == 'present' and (not exist):
            self.cli_add_command(view_cmd)
            self.cli_add_command(cmd)
            self.cli_add_command('quit')
        elif self.state == 'absent' and exist:
            self.cli_add_command(view_cmd)
            self.cli_add_command(cmd, undo=True)
            self.cli_add_command('quit')

    def config_vbdif(self):
        """configure command at the VBDIF interface view"""
        if not self.vbdif_name:
            return
        vbdif_cmd = 'interface %s' % self.vbdif_name.lower().capitalize()
        exist = is_config_exist(self.config, vbdif_cmd)
        if not exist:
            self.module.fail_json(msg='Error: Interface %s is not exist.' % self.vbdif_name)
        vbdif_view = False
        if self.vbdif_bind_vpn:
            cmd = 'ip binding vpn-instance %s' % self.vbdif_bind_vpn
            exist = is_config_exist(self.config, cmd)
            if self.state == 'present' and (not exist):
                if not vbdif_view:
                    self.cli_add_command(vbdif_cmd)
                    vbdif_view = True
                self.cli_add_command(cmd)
            elif self.state == 'absent' and exist:
                if not vbdif_view:
                    self.cli_add_command(vbdif_cmd)
                    vbdif_view = True
                self.cli_add_command(cmd, undo=True)
        if self.arp_distribute_gateway:
            cmd = 'arp distribute-gateway enable'
            exist = is_config_exist(self.config, cmd)
            if self.arp_distribute_gateway == 'enable' and (not exist):
                if not vbdif_view:
                    self.cli_add_command(vbdif_cmd)
                    vbdif_view = True
                self.cli_add_command(cmd)
            elif self.arp_distribute_gateway == 'disable' and exist:
                if not vbdif_view:
                    self.cli_add_command(vbdif_cmd)
                    vbdif_view = True
                self.cli_add_command(cmd, undo=True)
        if self.arp_direct_route:
            cmd = 'arp direct-route enable'
            exist = is_config_exist(self.config, cmd)
            if self.arp_direct_route == 'enable' and (not exist):
                if not vbdif_view:
                    self.cli_add_command(vbdif_cmd)
                    vbdif_view = True
                self.cli_add_command(cmd)
            elif self.arp_direct_route == 'disable' and exist:
                if not vbdif_view:
                    self.cli_add_command(vbdif_cmd)
                    vbdif_view = True
                self.cli_add_command(cmd, undo=True)
        if self.vbdif_mac:
            cmd = 'mac-address %s' % self.vbdif_mac
            exist = is_config_exist(self.config, cmd)
            if self.state == 'present' and (not exist):
                if not vbdif_view:
                    self.cli_add_command(vbdif_cmd)
                    vbdif_view = True
                self.cli_add_command(cmd)
            elif self.state == 'absent' and exist:
                if not vbdif_view:
                    self.cli_add_command(vbdif_cmd)
                    vbdif_view = True
                self.cli_add_command('undo mac-address')
        if vbdif_view:
            self.cli_add_command('quit')

    def is_valid_vbdif(self, ifname):
        """check is interface vbdif"""
        if not ifname.upper().startswith('VBDIF'):
            return False
        bdid = self.vbdif_name.replace(' ', '').upper().replace('VBDIF', '')
        if not bdid.isdigit():
            return False
        if int(bdid) < 1 or int(bdid) > 16777215:
            return False
        return True

    def is_valid_ip_vpn(self, vpname):
        """check ip vpn"""
        if not vpname:
            return False
        if vpname == '_public_':
            self.module.fail_json(msg='Error: The value C(_public_) is reserved and cannot be used as the VPN instance name.')
        if len(vpname) < 1 or len(vpname) > 31:
            self.module.fail_json(msg='Error: IP vpn name length is not in the range from 1 to 31.')
        return True

    def check_params(self):
        """Check all input params"""
        if self.dfs_id:
            if not self.dfs_id.isdigit():
                self.module.fail_json(msg='Error: DFS id is not digit.')
            if int(self.dfs_id) != 1:
                self.module.fail_json(msg='Error: DFS is not 1.')
        if self.dfs_source_ip:
            if not is_valid_v4addr(self.dfs_source_ip):
                self.module.fail_json(msg='Error: dfs_source_ip is invalid.')
            if self.dfs_source_vpn and (not self.is_valid_ip_vpn(self.dfs_source_vpn)):
                self.module.fail_json(msg='Error: dfs_source_vpn is invalid.')
        if self.dfs_source_vpn and (not self.dfs_source_ip):
            self.module.fail_json(msg='Error: dfs_source_vpn and dfs_source_ip must set at the same time.')
        if self.dfs_udp_port:
            if not self.dfs_udp_port.isdigit():
                self.module.fail_json(msg='Error: dfs_udp_port id is not digit.')
            if int(self.dfs_udp_port) < 1025 or int(self.dfs_udp_port) > 65535:
                self.module.fail_json(msg='dfs_udp_port is not ranges from 1025 to 65535.')
        if self.dfs_peer_ip:
            if not is_valid_v4addr(self.dfs_peer_ip):
                self.module.fail_json(msg='Error: dfs_peer_ip is invalid.')
            if self.dfs_peer_vpn and (not self.is_valid_ip_vpn(self.dfs_peer_vpn)):
                self.module.fail_json(msg='Error: dfs_peer_vpn is invalid.')
        if self.dfs_peer_vpn and (not self.dfs_peer_ip):
            self.module.fail_json(msg='Error: dfs_peer_vpn and dfs_peer_ip must set at the same time.')
        if self.vpn_instance and (not self.is_valid_ip_vpn(self.vpn_instance)):
            self.module.fail_json(msg='Error: vpn_instance is invalid.')
        if self.vpn_vni:
            if not self.vpn_vni.isdigit():
                self.module.fail_json(msg='Error: vpn_vni id is not digit.')
            if int(self.vpn_vni) < 1 or int(self.vpn_vni) > 16000000:
                self.module.fail_json(msg='vpn_vni is not  ranges from 1 to 16000000.')
        if bool(self.vpn_instance) != bool(self.vpn_vni):
            self.module.fail_json(msg='Error: vpn_instance and vpn_vni must set at the same time.')
        if self.vbdif_name:
            self.vbdif_name = self.vbdif_name.replace(' ', '').lower().capitalize()
            if not self.is_valid_vbdif(self.vbdif_name):
                self.module.fail_json(msg='Error: vbdif_name is invalid.')
        if self.vbdif_mac:
            mac = mac_format(self.vbdif_mac)
            if not mac:
                self.module.fail_json(msg='Error: vbdif_mac is invalid.')
            self.vbdif_mac = mac
        if self.vbdif_bind_vpn and (not self.is_valid_ip_vpn(self.vbdif_bind_vpn)):
            self.module.fail_json(msg='Error: vbdif_bind_vpn is invalid.')
        if self.dfs_id:
            if self.vpn_vni or self.arp_distribute_gateway == 'enable':
                self.module.fail_json(msg='Error: All-Active Gateways or Distributed Gateway config can not set at the same time.')

    def get_proposed(self):
        """get proposed info"""
        if self.dfs_id:
            self.proposed['dfs_id'] = self.dfs_id
            self.proposed['dfs_source_ip'] = self.dfs_source_ip
            self.proposed['dfs_source_vpn'] = self.dfs_source_vpn
            self.proposed['dfs_udp_port'] = self.dfs_udp_port
            self.proposed['dfs_all_active'] = self.dfs_all_active
            self.proposed['dfs_peer_ip'] = self.dfs_peer_ip
            self.proposed['dfs_peer_vpn'] = self.dfs_peer_vpn
        if self.vpn_instance:
            self.proposed['vpn_instance'] = self.vpn_instance
            self.proposed['vpn_vni'] = self.vpn_vni
        if self.vbdif_name:
            self.proposed['vbdif_name'] = self.vbdif_name
            self.proposed['vbdif_mac'] = self.vbdif_mac
            self.proposed['vbdif_bind_vpn'] = self.vbdif_bind_vpn
            self.proposed['arp_distribute_gateway'] = self.arp_distribute_gateway
            self.proposed['arp_direct_route'] = self.arp_direct_route
        self.proposed['state'] = self.state

    def get_existing(self):
        """get existing info"""
        if not self.config:
            return
        if is_config_exist(self.config, 'dfs-group 1'):
            self.existing['dfs_id'] = '1'
            self.existing['dfs_source_ip'] = get_dfs_source_ip(self.config)
            self.existing['dfs_source_vpn'] = get_dfs_source_vpn(self.config)
            self.existing['dfs_udp_port'] = get_dfs_udp_port(self.config)
            if is_config_exist(self.config, 'active-active-gateway'):
                self.existing['dfs_all_active'] = 'enable'
                self.existing['dfs_peers'] = get_dfs_peers(self.config)
            else:
                self.existing['dfs_all_active'] = 'disable'
        if self.vpn_instance:
            self.existing['vpn_instance'] = get_ip_vpn(self.config)
            self.existing['vpn_vni'] = get_ip_vpn_vni(self.config)
        if self.vbdif_name:
            self.existing['vbdif_name'] = self.vbdif_name
            self.existing['vbdif_mac'] = get_vbdif_mac(self.config)
            self.existing['vbdif_bind_vpn'] = get_vbdif_vpn(self.config)
            if is_config_exist(self.config, 'arp distribute-gateway enable'):
                self.existing['arp_distribute_gateway'] = 'enable'
            else:
                self.existing['arp_distribute_gateway'] = 'disable'
            if is_config_exist(self.config, 'arp direct-route enable'):
                self.existing['arp_direct_route'] = 'enable'
            else:
                self.existing['arp_direct_route'] = 'disable'

    def get_end_state(self):
        """get end state info"""
        config = self.get_current_config()
        if not config:
            return
        if is_config_exist(config, 'dfs-group 1'):
            self.end_state['dfs_id'] = '1'
            self.end_state['dfs_source_ip'] = get_dfs_source_ip(config)
            self.end_state['dfs_source_vpn'] = get_dfs_source_vpn(config)
            self.end_state['dfs_udp_port'] = get_dfs_udp_port(config)
            if is_config_exist(config, 'active-active-gateway'):
                self.end_state['dfs_all_active'] = 'enable'
                self.end_state['dfs_peers'] = get_dfs_peers(config)
            else:
                self.end_state['dfs_all_active'] = 'disable'
        if self.vpn_instance:
            self.end_state['vpn_instance'] = get_ip_vpn(config)
            self.end_state['vpn_vni'] = get_ip_vpn_vni(config)
        if self.vbdif_name:
            self.end_state['vbdif_name'] = self.vbdif_name
            self.end_state['vbdif_mac'] = get_vbdif_mac(config)
            self.end_state['vbdif_bind_vpn'] = get_vbdif_vpn(config)
            if is_config_exist(config, 'arp distribute-gateway enable'):
                self.end_state['arp_distribute_gateway'] = 'enable'
            else:
                self.end_state['arp_distribute_gateway'] = 'disable'
            if is_config_exist(config, 'arp direct-route enable'):
                self.end_state['arp_direct_route'] = 'enable'
            else:
                self.end_state['arp_direct_route'] = 'disable'

    def work(self):
        """worker"""
        self.check_params()
        self.config = self.get_current_config()
        self.get_existing()
        self.get_proposed()
        if self.dfs_id:
            self.config_dfs_group()
        if self.vpn_instance:
            self.config_ip_vpn()
        if self.vbdif_name:
            self.config_vbdif()
        if self.commands:
            self.cli_load_config(self.commands)
            self.changed = True
        self.get_end_state()
        self.results['changed'] = self.changed
        self.results['proposed'] = self.proposed
        self.results['existing'] = self.existing
        self.results['end_state'] = self.end_state
        if self.changed:
            self.results['updates'] = self.updates_cmd
        else:
            self.results['updates'] = list()
        self.module.exit_json(**self.results)