from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def _get_common_cluster_args(self):
    args = {'clustername': self.module.params.get('name'), 'hypervisor': self.module.params.get('hypervisor'), 'clustertype': self.module.params.get('cluster_type')}
    state = self.module.params.get('state')
    if state in ['enabled', 'disabled']:
        args['allocationstate'] = state.capitalize()
    return args