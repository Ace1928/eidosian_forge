from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
class ShowZonesetActive(object):
    """docstring for ShowZonesetActive"""

    def __init__(self, module, vsan):
        self.vsan = vsan
        self.module = module
        self.activeZSName = None
        self.parseCmdOutput()

    def execute_show_zoneset_active_cmd(self):
        command = 'show zoneset active vsan ' + str(self.vsan) + ' | grep zoneset'
        output = execute_show_command(command, self.module)[0]
        return output

    def parseCmdOutput(self):
        patZoneset = 'zoneset name (\\S+) vsan ' + str(self.vsan)
        output = self.execute_show_zoneset_active_cmd().split('\n')
        if len(output) == 0:
            return
        else:
            for line in output:
                line = line.strip()
                mzs = re.match(patZoneset, line.strip())
                if mzs:
                    self.activeZSName = mzs.group(1).strip()
                    return

    def isZonesetActive(self, zsname):
        if zsname == self.activeZSName:
            return True
        return False