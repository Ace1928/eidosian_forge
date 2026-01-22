from __future__ import absolute_import, division, print_function
import string
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
class showDeviceAliasStatus(object):
    """docstring for showDeviceAliasStatus"""

    def __init__(self, module):
        self.module = module
        self.distribute = ''
        self.mode = ''
        self.locked = False
        self.update()

    def execute_show_cmd(self, cmd):
        output = execute_show_command(cmd, self.module)[0]
        return output

    def update(self):
        command = 'show device-alias status'
        output = self.execute_show_cmd(command).split('\n')
        for o in output:
            if 'Fabric Distribution' in o:
                self.distribute = o.split(':')[1].strip().lower()
            if 'Mode' in o:
                self.mode = o.split('Mode:')[1].strip().lower()
            if 'Locked' in o:
                self.locked = True

    def isLocked(self):
        return self.locked

    def getDistribute(self):
        return self.distribute

    def getMode(self):
        return self.mode