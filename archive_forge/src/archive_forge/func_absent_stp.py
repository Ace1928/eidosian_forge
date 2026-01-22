from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import exec_command, load_config, ce_argument_spec
def absent_stp(self):
    """ Absent stp configuration """
    cmds = list()
    if self.stp_mode:
        if self.stp_mode == self.cur_cfg['stp_mode']:
            if self.stp_mode != 'mstp':
                cmd = 'undo stp mode'
                cmds.append(cmd)
                self.updates_cmd.append(cmd)
                self.changed = True
    if self.stp_enable:
        if self.stp_enable != self.cur_cfg['stp_enable']:
            cmd = 'stp %s' % self.stp_enable
            cmds.append(cmd)
            self.updates_cmd.append(cmd)
    if self.stp_converge:
        if self.stp_converge == self.cur_cfg['stp_converge']:
            cmd = 'undo stp converge'
            cmds.append(cmd)
            self.updates_cmd.append(cmd)
            self.changed = True
    if self.edged_port:
        if self.interface == 'all':
            if self.edged_port != self.cur_cfg['edged_port']:
                if self.edged_port == 'enable':
                    cmd = 'stp edged-port default'
                    cmds.append(cmd)
                    self.updates_cmd.append(cmd)
                else:
                    cmd = 'undo stp edged-port default'
                    cmds.append(cmd)
                    self.updates_cmd.append(cmd)
    if self.bpdu_filter:
        if self.interface == 'all':
            if self.bpdu_filter != self.cur_cfg['bpdu_filter']:
                if self.bpdu_filter == 'enable':
                    cmd = 'stp bpdu-filter default'
                    cmds.append(cmd)
                    self.updates_cmd.append(cmd)
                else:
                    cmd = 'undo stp bpdu-filter default'
                    cmds.append(cmd)
                    self.updates_cmd.append(cmd)
    if self.bpdu_protection:
        if self.bpdu_protection != self.cur_cfg['bpdu_protection']:
            if self.bpdu_protection == 'enable':
                cmd = 'stp bpdu-protection'
                cmds.append(cmd)
                self.updates_cmd.append(cmd)
            else:
                cmd = 'undo stp bpdu-protection'
                cmds.append(cmd)
                self.updates_cmd.append(cmd)
    if self.tc_protection:
        if self.tc_protection != self.cur_cfg['tc_protection']:
            if self.tc_protection == 'enable':
                cmd = 'stp tc-protection'
                cmds.append(cmd)
                self.updates_cmd.append(cmd)
            else:
                cmd = 'undo stp tc-protection'
                cmds.append(cmd)
                self.updates_cmd.append(cmd)
    if self.tc_protection_interval:
        if self.tc_protection_interval == self.cur_cfg['tc_protection_interval']:
            cmd = 'undo stp tc-protection interval'
            cmds.append(cmd)
            self.updates_cmd.append(cmd)
            self.changed = True
    if self.tc_protection_threshold:
        if self.tc_protection_threshold == self.cur_cfg['tc_protection_threshold']:
            if self.tc_protection_threshold != '1':
                cmd = 'undo stp tc-protection threshold'
                cmds.append(cmd)
                self.updates_cmd.append(cmd)
                self.changed = True
    if self.interface and self.interface != 'all':
        tmp_changed = False
        cmd = 'interface %s' % self.interface
        cmds.append(cmd)
        self.updates_cmd.append(cmd)
        if self.edged_port:
            if self.edged_port != self.cur_cfg['edged_port']:
                if self.edged_port == 'enable':
                    cmd = 'stp edged-port enable'
                    cmds.append(cmd)
                    self.updates_cmd.append(cmd)
                    tmp_changed = True
                else:
                    cmd = 'undo stp edged-port'
                    cmds.append(cmd)
                    self.updates_cmd.append(cmd)
                    tmp_changed = True
        if self.bpdu_filter:
            if self.bpdu_filter != self.cur_cfg['bpdu_filter']:
                if self.bpdu_filter == 'enable':
                    cmd = 'stp bpdu-filter enable'
                    cmds.append(cmd)
                    self.updates_cmd.append(cmd)
                    tmp_changed = True
                else:
                    cmd = 'undo stp bpdu-filter'
                    cmds.append(cmd)
                    self.updates_cmd.append(cmd)
                    tmp_changed = True
        if self.root_protection:
            if self.root_protection == 'enable' and self.cur_cfg['loop_protection'] == 'enable':
                self.module.fail_json(msg='Error: The interface has enable loop_protection, can not enable root_protection.')
            if self.root_protection != self.cur_cfg['root_protection']:
                if self.root_protection == 'enable':
                    cmd = 'stp root-protection'
                    cmds.append(cmd)
                    self.updates_cmd.append(cmd)
                    tmp_changed = True
                else:
                    cmd = 'undo stp root-protection'
                    cmds.append(cmd)
                    self.updates_cmd.append(cmd)
                    tmp_changed = True
        if self.loop_protection:
            if self.loop_protection == 'enable' and self.cur_cfg['root_protection'] == 'enable':
                self.module.fail_json(msg='Error: The interface has enable root_protection, can not enable loop_protection.')
            if self.loop_protection != self.cur_cfg['loop_protection']:
                if self.loop_protection == 'enable':
                    cmd = 'stp loop-protection'
                    cmds.append(cmd)
                    self.updates_cmd.append(cmd)
                    tmp_changed = True
                else:
                    cmd = 'undo stp loop-protection'
                    cmds.append(cmd)
                    self.updates_cmd.append(cmd)
                    tmp_changed = True
        if self.cost:
            if self.cost == self.cur_cfg['cost']:
                cmd = 'undo stp cost'
                cmds.append(cmd)
                self.updates_cmd.append(cmd)
                tmp_changed = True
        if not tmp_changed:
            cmd = 'interface %s' % self.interface
            self.updates_cmd.remove(cmd)
            cmds.remove(cmd)
    if cmds:
        self.cli_load_config(cmds)
        self.changed = True