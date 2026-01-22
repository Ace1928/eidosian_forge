from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (
def _update_network(self):
    network = self.get_physical_network()
    args = dict(id=network['id'])
    args.update(self._get_common_args())
    if self.has_changed(args, network):
        self.result['changed'] = True
        if not self.module.check_mode:
            resource = self.query_api('updatePhysicalNetwork', **args)
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                self.physical_network = self.poll_job(resource, 'physicalnetwork')
    return self.physical_network