from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
class AnsibleCloudStackInstanceGroup(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackInstanceGroup, self).__init__(module)
        self.instance_group = None

    def get_instance_group(self):
        if self.instance_group:
            return self.instance_group
        name = self.module.params.get('name')
        args = {'account': self.get_account('name'), 'domainid': self.get_domain('id'), 'projectid': self.get_project('id'), 'fetch_list': True}
        instance_groups = self.query_api('listInstanceGroups', **args)
        if instance_groups:
            for g in instance_groups:
                if name in [g['name'], g['id']]:
                    self.instance_group = g
                    break
        return self.instance_group

    def present_instance_group(self):
        instance_group = self.get_instance_group()
        if not instance_group:
            self.result['changed'] = True
            args = {'name': self.module.params.get('name'), 'account': self.get_account('name'), 'domainid': self.get_domain('id'), 'projectid': self.get_project('id')}
            if not self.module.check_mode:
                res = self.query_api('createInstanceGroup', **args)
                instance_group = res['instancegroup']
        return instance_group

    def absent_instance_group(self):
        instance_group = self.get_instance_group()
        if instance_group:
            self.result['changed'] = True
            if not self.module.check_mode:
                self.query_api('deleteInstanceGroup', id=instance_group['id'])
        return instance_group