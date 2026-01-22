from __future__ import (absolute_import, division, print_function)
import re
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, load_config
from ansible.module_utils.connection import exec_command
class Mtu(object):
    """set mtu"""

    def __init__(self, argument_spec):
        self.spec = argument_spec
        self.module = None
        self.init_module()
        self.interface = self.module.params['interface']
        self.mtu = self.module.params['mtu']
        self.state = self.module.params['state']
        self.jbf_max = self.module.params['jumbo_max'] or None
        self.jbf_min = self.module.params['jumbo_min'] or None
        self.jbf_config = list()
        self.jbf_cli = ''
        self.commands = list()
        self.changed = False
        self.updates_cmd = list()
        self.results = dict()
        self.proposed = dict()
        self.existing = dict()
        self.end_state = dict()
        self.intf_info = dict()
        self.intf_type = None

    def init_module(self):
        """ init_module"""
        self.module = AnsibleModule(argument_spec=self.spec, supports_check_mode=True)

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

    def get_interface_dict(self, ifname):
        """ get one interface attributes dict."""
        intf_info = dict()
        flags = list()
        exp = '| ignore-case section include ^#\\s+interface %s\\s+' % ifname.replace(' ', '')
        flags.append(exp)
        output = self.get_config(flags)
        output_list = output.split('\n')
        if output_list is None:
            return intf_info
        mtu = None
        for config in output_list:
            config = config.strip()
            if config.startswith('mtu'):
                mtu = re.findall('.*mtu\\s*([0-9]*)', output)[0]
        intf_info = dict(ifName=ifname, ifMtu=mtu)
        return intf_info

    def prase_jumboframe_para(self, config_str):
        """prase_jumboframe_para"""
        interface_cli = 'interface %s' % self.interface.replace(' ', '').lower()
        if config_str.find(interface_cli) == -1:
            self.module.fail_json(msg='Error: Interface does not exist.')
        try:
            npos1 = config_str.index('jumboframe enable')
        except ValueError:
            return [9216, 1518]
        try:
            npos2 = config_str.index('\n', npos1)
            config_str_tmp = config_str[npos1:npos2]
        except ValueError:
            config_str_tmp = config_str[npos1:]
        return re.findall('([0-9]+)', config_str_tmp)

    def cli_load_config(self):
        """load config by cli"""
        if not self.module.check_mode:
            if len(self.commands) > 1:
                load_config(self.module, self.commands)
                self.changed = True

    def cli_add_command(self, command, undo=False):
        """add command to self.update_cmd and self.commands"""
        if undo and command.lower() not in ['quit', 'return']:
            cmd = 'undo ' + command
        else:
            cmd = command
        self.commands.append(cmd)

    def get_jumboframe_config(self):
        """ get_jumboframe_config"""
        flags = list()
        exp = '| ignore-case section include ^#\\s+interface %s\\s+' % self.interface.replace(' ', '')
        flags.append(exp)
        output = self.get_config(flags)
        output = output.replace('*', '').lower()
        return self.prase_jumboframe_para(output)

    def set_jumboframe(self):
        """ set_jumboframe"""
        if self.state == 'present':
            if not self.jbf_max and (not self.jbf_min):
                return
            jbf_value = self.get_jumboframe_config()
            self.jbf_config = copy.deepcopy(jbf_value)
            if len(jbf_value) == 1:
                jbf_value.append('1518')
                self.jbf_config.append('1518')
            if not self.jbf_max:
                return
            if len(jbf_value) > 2 or len(jbf_value) == 0:
                self.module.fail_json(msg='Error: Get jubmoframe config value num error.')
            if self.jbf_min is None:
                if jbf_value[0] == self.jbf_max:
                    return
            elif jbf_value[0] == self.jbf_max and jbf_value[1] == self.jbf_min:
                return
            if jbf_value[0] != self.jbf_max:
                jbf_value[0] = self.jbf_max
            if jbf_value[1] != self.jbf_min and self.jbf_min is not None:
                jbf_value[1] = self.jbf_min
            else:
                jbf_value.pop(1)
        else:
            jbf_value = self.get_jumboframe_config()
            self.jbf_config = copy.deepcopy(jbf_value)
            if jbf_value == [9216, 1518]:
                return
            jbf_value = [9216, 1518]
        if len(jbf_value) == 2:
            self.jbf_cli = 'jumboframe enable %s %s' % (jbf_value[0], jbf_value[1])
        else:
            self.jbf_cli = 'jumboframe enable %s' % jbf_value[0]
        self.cli_add_command(self.jbf_cli)
        if self.state == 'present':
            if self.jbf_min:
                self.updates_cmd.append('jumboframe enable %s %s' % (self.jbf_max, self.jbf_min))
            else:
                self.updates_cmd.append('jumboframe enable %s' % self.jbf_max)
        else:
            self.updates_cmd.append('undo jumboframe enable')
        return

    def merge_interface(self, ifname, mtu):
        """ Merge interface mtu."""
        xmlstr = ''
        change = False
        command = 'interface %s' % ifname
        self.cli_add_command(command)
        if self.state == 'present':
            if mtu and self.intf_info['ifMtu'] != mtu:
                command = 'mtu %s' % mtu
                self.cli_add_command(command)
                self.updates_cmd.append('mtu %s' % mtu)
                change = True
        elif self.intf_info['ifMtu'] != '1500' and self.intf_info['ifMtu']:
            command = 'mtu 1500'
            self.cli_add_command(command)
            self.updates_cmd.append('undo mtu')
            change = True
        return

    def check_params(self):
        """Check all input params"""
        if self.interface:
            self.intf_type = get_interface_type(self.interface)
            if not self.intf_type:
                self.module.fail_json(msg='Error: Interface name of %s is error.' % self.interface)
        if not self.intf_type:
            self.module.fail_json(msg='Error: Interface %s is error.')
        if self.mtu:
            if not self.mtu.isdigit():
                self.module.fail_json(msg='Error: Mtu is invalid.')
            if int(self.mtu) < 46 or int(self.mtu) > 9600:
                self.module.fail_json(msg='Error: Mtu is not in the range from 46 to 9600.')
        self.intf_info = self.get_interface_dict(self.interface)
        if not self.intf_info:
            self.module.fail_json(msg='Error: interface does not exist.')
        if self.state == 'present':
            if self.jbf_max:
                if not is_interface_support_setjumboframe(self.interface):
                    self.module.fail_json(msg='Error: Interface %s does not support jumboframe set.' % self.interface)
                if not self.jbf_max.isdigit():
                    self.module.fail_json(msg='Error: Max jumboframe is not digit.')
                if int(self.jbf_max) > 12288 or int(self.jbf_max) < 1536:
                    self.module.fail_json(msg='Error: Max jumboframe is between 1536 to 12288.')
            if self.jbf_min:
                if not self.jbf_min.isdigit():
                    self.module.fail_json(msg='Error: Min jumboframe is not digit.')
                if not self.jbf_max:
                    self.module.fail_json(msg='Error: please specify max jumboframe value.')
                if int(self.jbf_min) > int(self.jbf_max) or int(self.jbf_min) < 1518:
                    self.module.fail_json(msg='Error: Min jumboframe is between 1518 to jumboframe max value.')
            if self.jbf_min is not None:
                if self.jbf_max is None:
                    self.module.fail_json(msg='Error: please input MAX jumboframe value.')

    def get_proposed(self):
        """ get_proposed"""
        self.proposed['state'] = self.state
        if self.interface:
            self.proposed['interface'] = self.interface
        if self.state == 'present':
            if self.mtu:
                self.proposed['mtu'] = self.mtu
            if self.jbf_max:
                if self.jbf_min:
                    self.proposed['jumboframe'] = 'jumboframe enable %s %s' % (self.jbf_max, self.jbf_min)
                else:
                    self.proposed['jumboframe'] = 'jumboframe enable %s %s' % (self.jbf_max, 1518)

    def get_existing(self):
        """ get_existing"""
        if self.intf_info:
            self.existing['interface'] = self.intf_info['ifName']
            self.existing['mtu'] = self.intf_info['ifMtu']
        if self.intf_info:
            if not self.existing['interface']:
                self.existing['interface'] = self.interface
            if len(self.jbf_config) != 2:
                return
            self.existing['jumboframe'] = 'jumboframe enable %s %s' % (self.jbf_config[0], self.jbf_config[1])

    def get_end_state(self):
        """ get_end_state"""
        if self.intf_info:
            end_info = self.get_interface_dict(self.interface)
            if end_info:
                self.end_state['interface'] = end_info['ifName']
                self.end_state['mtu'] = end_info['ifMtu']
        if self.intf_info:
            if not self.end_state['interface']:
                self.end_state['interface'] = self.interface
            if self.state == 'absent':
                self.end_state['jumboframe'] = 'jumboframe enable %s %s' % (9216, 1518)
            elif not self.jbf_max and (not self.jbf_min):
                if len(self.jbf_config) != 2:
                    return
                self.end_state['jumboframe'] = 'jumboframe enable %s %s' % (self.jbf_config[0], self.jbf_config[1])
            elif self.jbf_min:
                self.end_state['jumboframe'] = 'jumboframe enable %s %s' % (self.jbf_max, self.jbf_min)
            else:
                self.end_state['jumboframe'] = 'jumboframe enable %s %s' % (self.jbf_max, 1518)
        if self.end_state == self.existing:
            self.changed = False

    def work(self):
        """worker"""
        self.check_params()
        self.get_proposed()
        self.merge_interface(self.interface, self.mtu)
        self.set_jumboframe()
        self.cli_load_config()
        self.get_existing()
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