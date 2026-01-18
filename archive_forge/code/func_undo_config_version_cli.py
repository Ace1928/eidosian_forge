from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, \
def undo_config_version_cli(self):
    """ Undo config version by cli """
    if 'disable' in self.cur_cli_cfg['version']:
        pass
    else:
        cmd = 'snmp-agent sys-info version  %s disable' % self.cur_cli_cfg['version']
        cmds = list()
        cmds.append(cmd)
        self.updates_cmd.append(cmd)
        self.cli_load_config(cmds)
        self.changed = True