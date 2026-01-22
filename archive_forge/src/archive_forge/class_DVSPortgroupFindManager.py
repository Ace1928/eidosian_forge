from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import (
from ansible.module_utils.six.moves.urllib.parse import unquote
class DVSPortgroupFindManager(PyVmomi):

    def __init__(self, module):
        super(DVSPortgroupFindManager, self).__init__(module)
        self.dvs_name = self.params['dvswitch']
        self.vlan = self.params['vlanid']
        self.cmp_vlans = True if self.vlan else False
        self.pgs = self.find_portgroups_by_name(self.content, self.module.params['name'])
        if self.dvs_name:
            self.pgs = self.find_portgroups_by_dvs(self.pgs, self.dvs_name)

    def find_portgroups_by_name(self, content, name=None):
        vimtype = [vim.dvs.DistributedVirtualPortgroup]
        container = content.viewManager.CreateContainerView(content.rootFolder, vimtype, True)
        if not name:
            obj = container.view
        else:
            obj = []
            for c in container.view:
                if name in c.name:
                    obj.append(c)
        return obj

    def find_portgroups_by_dvs(self, pgl, dvs):
        obj = []
        for c in pgl:
            if dvs in c.config.distributedVirtualSwitch.name:
                obj.append(c)
        return obj

    def vlan_match(self, pgup, userup, vlanlst):
        res = False
        if pgup and userup:
            return True
        for ln in vlanlst:
            if '-' in ln:
                arr = ln.split('-')
                if int(arr[0]) < self.vlan and self.vlan < int(arr[1]):
                    res = True
            elif ln == str(self.vlan):
                res = True
        return res

    def get_dvs_portgroup(self):
        pgroups = self.pgs
        pglist = []
        for pg in pgroups:
            trunk = False
            pvlan = False
            vlanInfo = pg.config.defaultPortConfig.vlan
            cl1 = vim.dvs.VmwareDistributedVirtualSwitch.TrunkVlanSpec
            cl2 = vim.dvs.VmwareDistributedVirtualSwitch.PvlanSpec
            vlan_id_list = []
            if isinstance(vlanInfo, cl1):
                trunk = True
                for item in vlanInfo.vlanId:
                    if item.start == item.end:
                        vlan_id_list.append(str(item.start))
                    else:
                        vlan_id_list.append(str(item.start) + '-' + str(item.end))
            elif isinstance(vlanInfo, cl2):
                pvlan = True
                vlan_id_list.append(str(vlanInfo.pvlanId))
            else:
                vlan_id_list.append(str(vlanInfo.vlanId))
            if self.cmp_vlans:
                if self.vlan_match(pg.config.uplink, self.module.params['show_uplink'], vlan_id_list):
                    pglist.append(dict(name=unquote(pg.name), trunk=trunk, pvlan=pvlan, vlan_id=','.join(vlan_id_list), dvswitch=pg.config.distributedVirtualSwitch.name))
            else:
                pglist.append(dict(name=unquote(pg.name), trunk=trunk, pvlan=pvlan, vlan_id=','.join(vlan_id_list), dvswitch=pg.config.distributedVirtualSwitch.name))
        return pglist