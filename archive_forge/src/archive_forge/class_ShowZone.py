from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
class ShowZone(object):
    """docstring for ShowZone"""

    def __init__(self, module, vsan):
        self.vsan = vsan
        self.module = module
        self.zDetails = {}
        self.parseCmdOutput()

    def execute_show_zone_vsan_cmd(self):
        command = 'show zone vsan ' + str(self.vsan)
        output = execute_show_command(command, self.module)[0]
        return output

    def parseCmdOutput(self):
        patZone = 'zone name (\\S+) vsan ' + str(self.vsan)
        output = self.execute_show_zone_vsan_cmd().split('\n')
        for line in output:
            line = re.sub('[\\[].*?[\\]]', '', line)
            line = ' '.join(line.strip().split())
            if 'init' in line:
                line = line.replace('init', 'initiator')
            m = re.match(patZone, line)
            if m:
                zonename = m.group(1).strip()
                self.zDetails[zonename] = []
                continue
            elif 'pwwn' in line or 'device-alias' in line:
                v = self.zDetails[zonename]
                v.append(line)
                self.zDetails[zonename] = v

    def isZonePresent(self, zname):
        return zname in self.zDetails.keys()

    def isZoneMemberPresent(self, zname, cmd):
        if zname in self.zDetails.keys():
            zonememlist = self.zDetails[zname]
            for eachline in zonememlist:
                if cmd in eachline:
                    return True
        return False

    def get_zDetails(self):
        return self.zDetails