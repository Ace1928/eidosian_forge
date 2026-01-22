from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec, find_obj
class CustomAttributeManager(PyVmomi):

    def __init__(self, module):
        super(CustomAttributeManager, self).__init__(module)
        if not self.is_vcenter():
            self.module.fail_json(msg='You have to connect to a vCenter server!')
        object_types_map = {'Cluster': vim.ClusterComputeResource, 'Datacenter': vim.Datacenter, 'Datastore': vim.Datastore, 'DistributedVirtualPortgroup': vim.DistributedVirtualPortgroup, 'DistributedVirtualSwitch': vim.DistributedVirtualSwitch, 'Folder': vim.Folder, 'HostSystem': vim.HostSystem, 'ResourcePool': vim.ResourcePool, 'VirtualMachine': vim.VirtualMachine}
        self.object_type = object_types_map[self.params['object_type']]
        self.object_name = self.params['object_name']
        self.obj = find_obj(self.content, [self.object_type], self.params['object_name'])
        if self.obj is None:
            module.fail_json(msg='Unable to manage custom attributes for non-existing object %s.' % self.object_name)
        self.ca_list = self.params['custom_attributes'].copy()
        for ca in self.ca_list:
            for av_field in self.obj.availableField:
                if av_field.name == ca['name']:
                    ca['key'] = av_field.key
                    break
        for ca in self.ca_list:
            if 'key' not in ca:
                self.module.fail_json(msg='Custom attribute %s does not exist for object type %s.' % (ca['name'], self.params['object_type']))

    def set_custom_attributes(self):
        changed = False
        obj_cas_set = [x.key for x in self.obj.value]
        for ca in self.ca_list:
            if ca['key'] not in obj_cas_set:
                changed = True
                if not self.module.check_mode:
                    self.content.customFieldsManager.SetField(entity=self.obj, key=ca['key'], value=ca['value'])
                continue
            for x in self.obj.customValue:
                if ca['key'] == x.key and ca['value'] != x.value:
                    changed = True
                    if not self.module.check_mode:
                        self.content.customFieldsManager.SetField(entity=self.obj, key=ca['key'], value=ca['value'])
        return {'changed': changed, 'failed': False}

    def remove_custom_attributes(self):
        changed = False
        for ca in self.ca_list:
            for x in self.obj.customValue:
                if ca['key'] == x.key and x.value != '':
                    changed = True
                    if not self.module.check_mode:
                        self.content.customFieldsManager.SetField(entity=self.obj, key=ca['key'], value='')
        return {'changed': changed, 'failed': False}