from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
class ShowZoneStatus(object):
    """docstring for ShowZoneStatus"""

    def __init__(self, module, vsan):
        self.vsan = vsan
        self.vsanAbsent = False
        self.module = module
        self.default_zone = ''
        self.mode = ''
        self.session = ''
        self.sz = ''
        self.locked = False
        self.update()

    def execute_show_zone_status_cmd(self):
        command = 'show zone status vsan ' + str(self.vsan)
        output = execute_show_command(command, self.module)[0]
        return output

    def update(self):
        output = self.execute_show_zone_status_cmd().split('\n')
        patfordefzone = 'VSAN: ' + str(self.vsan) + ' default-zone:\\s+(\\S+).*'
        patformode = '.*mode:\\s+(\\S+).*'
        patforsession = '^session:\\s+(\\S+).*'
        patforsz = '.*smart-zoning:\\s+(\\S+).*'
        for line in output:
            if 'is not configured' in line:
                self.vsanAbsent = True
                break
            mdefz = re.match(patfordefzone, line.strip())
            mmode = re.match(patformode, line.strip())
            msession = re.match(patforsession, line.strip())
            msz = re.match(patforsz, line.strip())
            if mdefz:
                self.default_zone = mdefz.group(1)
            if mmode:
                self.mode = mmode.group(1)
            if msession:
                self.session = msession.group(1)
                if self.session != 'none':
                    self.locked = True
            if msz:
                self.sz = msz.group(1)

    def isLocked(self):
        return self.locked

    def getDefaultZone(self):
        return self.default_zone

    def getMode(self):
        return self.mode

    def getSmartZoningStatus(self):
        return self.sz

    def isVsanAbsent(self):
        return self.vsanAbsent