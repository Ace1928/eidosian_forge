from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.vultr_v2 import AnsibleVultr, vultr_argument_spec
class AnsibleVultrObjectStorage(AnsibleVultr):

    def configure(self):
        super(AnsibleVultrObjectStorage, self).configure()
        cluster = self.get_cluster()
        self.module.params['cluster_id'] = cluster['id']
        self.module.params['region'] = cluster['region']

    def get_cluster(self):
        return self.query_filter_list_by_name(key_name='hostname', param_key='cluster', path='/object-storage/clusters', result_key='clusters', fail_not_found=True)

    def create_or_update(self):
        resource = super(AnsibleVultrObjectStorage, self).create_or_update()
        if resource:
            resource = self.wait_for_state(resource=resource, key='status', states=['active'])
        return resource