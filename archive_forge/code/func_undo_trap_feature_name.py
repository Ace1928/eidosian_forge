from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import load_config, ce_argument_spec, run_commands
from ansible.module_utils.connection import exec_command
def undo_trap_feature_name(self):
    """ Undo feature name for trap """
    if self.feature_name == 'all':
        cmd = 'undo snmp-agent trap enable'
    else:
        cmd = 'undo snmp-agent trap enable feature-name %s' % self.feature_name
        if self.trap_name:
            cmd += ' trap-name %s' % self.trap_name
    self.updates_cmd.append(cmd)
    cmds = list()
    cmds.append(cmd)
    self.cli_load_config(cmds)
    self.changed = True