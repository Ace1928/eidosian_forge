from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, vmware_argument_spec
def add_custom_def(self, field):
    changed = False
    found = False
    for x in self.custom_field_mgr:
        if x.name == field and x.managedObjectType == self.object_type:
            found = True
            break
    if not found:
        changed = True
        if not self.module.check_mode:
            self.content.customFieldsManager.AddFieldDefinition(name=field, moType=self.object_type)
    return {'changed': changed, 'failed': False}