from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_nc_config, set_nc_config, \
def undo_config_connect_port_cli(self):
    """ Undo config connect port by cli """
    if 'connect port' in self.cur_cli_cfg.keys():
        if not self.cur_cli_cfg['connect port']:
            pass
        else:
            cmd = 'undo snmp-agent udp-port'
            cmds = list()
            cmds.append(cmd)
            self.updates_cmd.append(cmd)
            connect_port = '161'
            conf_str = CE_MERGE_SNMP_PORT % connect_port
            self.netconf_set_config(conf_str=conf_str)
            self.changed = True