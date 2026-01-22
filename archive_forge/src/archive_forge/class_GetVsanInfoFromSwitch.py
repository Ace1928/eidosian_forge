from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.nxos.plugins.module_utils.network.nxos.nxos import (
class GetVsanInfoFromSwitch(object):
    """docstring for GetVsanInfoFromSwitch"""

    def __init__(self, module):
        self.module = module
        self.vsaninfo = {}
        self.processShowVsan()
        self.processShowVsanMembership()

    def execute_show_vsan_cmd(self):
        output = execute_show_command('show vsan', self.module)[0]
        return output

    def execute_show_vsan_mem_cmd(self):
        output = execute_show_command('show vsan membership', self.module)[0]
        return output

    def processShowVsan(self):
        patv = '^vsan\\s+(\\d+)\\s+information'
        patnamestate = 'name:(.*)state:(.*)'
        patoperstate = 'operational state:(.*)'
        output = self.execute_show_vsan_cmd().split('\n')
        for o in output:
            z = re.match(patv, o.strip())
            if z:
                v = z.group(1).strip()
                self.vsaninfo[v] = Vsan(v)
            z1 = re.match(patnamestate, o.strip())
            if z1:
                n = z1.group(1).strip()
                s = z1.group(2).strip()
                self.vsaninfo[v].vsanname = n
                self.vsaninfo[v].vsanstate = s
            z2 = re.match(patoperstate, o.strip())
            if z2:
                oper = z2.group(1).strip()
                self.vsaninfo[v].vsanoperstate = oper
        self.vsaninfo['4079'] = Vsan('4079')
        self.vsaninfo['4094'] = Vsan('4094')

    def processShowVsanMembership(self):
        patv = '^vsan\\s+(\\d+).*'
        output = self.execute_show_vsan_mem_cmd().split('\n')
        memlist = []
        v = None
        for o in output:
            z = re.match(patv, o.strip())
            if z:
                if v is not None:
                    self.vsaninfo[v].vsaninterfaces = memlist
                    memlist = []
                v = z.group(1)
            if 'interfaces' not in o:
                llist = o.strip().split()
                memlist = memlist + llist
        self.vsaninfo[v].vsaninterfaces = memlist

    def getVsanInfoObjects(self):
        return self.vsaninfo