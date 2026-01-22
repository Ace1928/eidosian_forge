from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ngine_io.cloudstack.plugins.module_utils.cloudstack import (
def absent_record(self):
    instance = self.get_instance()
    if instance:
        record = self.get_record(instance)
        if record:
            self.result['diff']['before'] = record
            self.result['changed'] = True
            if not self.module.check_mode:
                self.query_api('deleteReverseDnsFromVirtualMachine', id=instance['id'])
        return record