from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.vultr_v2 import AnsibleVultr, vultr_argument_spec
class AnsibleVultrDnsRecord(AnsibleVultr):

    def query(self):
        multiple = self.module.params.get('multiple')
        name = self.module.params.get('name')
        data = self.module.params.get('data')
        record_type = self.module.params.get('type')
        result = dict()
        for resource in self.query_list():
            if resource.get('type') != record_type:
                continue
            if resource.get('name') == name:
                if not multiple:
                    if result:
                        self.module.fail_json(msg='More than one record with record_type=%s and name=%s params. Use multiple=true for more than one record.' % (record_type, name))
                    else:
                        result = resource
                elif resource.get('data') == data:
                    return resource
        return result