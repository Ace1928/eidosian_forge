from __future__ import (absolute_import, division, print_function)
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import load_config
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import get_connection, rm_config_prefix
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec
def absent_netstream(self):
    """ Absent netstream configuration """
    cmds = list()
    absent_netstream_attr = False
    if not self.netstream_cfg:
        return
    if self.description or self.match or self.collect_counter or self.collect_interface:
        absent_netstream_attr = True
    if absent_netstream_attr:
        if self.type == 'ip':
            cmd = 'netstream record %s ip' % self.record_name
        else:
            cmd = 'netstream record %s vxlan inner-ip' % self.record_name
        cmds.append(cmd)
        if self.description:
            cfg = 'description %s' % self.description
            if self.netstream_cfg and cfg in self.netstream_cfg:
                cmd = 'undo description %s' % self.description
                cmds.append(cmd)
                self.updates_cmd.append(cmd)
        if self.match:
            if self.type == 'ip':
                cfg = 'match ip %s' % self.match
            else:
                cfg = 'match inner-ip %s' % self.match
            if self.netstream_cfg and cfg in self.netstream_cfg:
                if self.type == 'ip':
                    cmd = 'undo match ip %s' % self.match
                else:
                    cmd = 'undo match inner-ip %s' % self.match
                cmds.append(cmd)
                self.updates_cmd.append(cmd)
        if self.collect_counter:
            cfg = 'collect counter %s' % self.collect_counter
            if self.netstream_cfg and cfg in self.netstream_cfg:
                cmd = 'undo collect counter %s' % self.collect_counter
                cmds.append(cmd)
                self.updates_cmd.append(cmd)
        if self.collect_interface:
            cfg = 'collect interface %s' % self.collect_interface
            if self.netstream_cfg and cfg in self.netstream_cfg:
                cmd = 'undo collect interface %s' % self.collect_interface
                cmds.append(cmd)
                self.updates_cmd.append(cmd)
        if len(cmds) > 1:
            self.cli_load_config(cmds)
            self.changed = True
    else:
        if self.type == 'ip':
            cmd = 'undo netstream record %s ip' % self.record_name
        else:
            cmd = 'undo netstream record %s vxlan inner-ip' % self.record_name
        cmds.append(cmd)
        self.updates_cmd.append(cmd)
        self.cli_load_config(cmds)
        self.changed = True