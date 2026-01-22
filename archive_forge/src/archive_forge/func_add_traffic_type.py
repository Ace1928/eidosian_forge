from __future__ import absolute_import, division, print_function
from ..module_utils.cloudstack import AnsibleCloudStack, cs_argument_spec, cs_required_together
from ansible.module_utils.basic import AnsibleModule
def add_traffic_type(self):
    traffic_type = self.module.params.get('traffic_type')
    args = {'physicalnetworkid': self.get_physical_network(key='id'), 'traffictype': traffic_type}
    args.update(self._get_additional_args())
    if not self.module.check_mode:
        resource = self.query_api('addTrafficType', **args)
        poll_async = self.module.params.get('poll_async')
        if poll_async:
            self.traffic_type = self.poll_job(resource, 'traffictype')
    return self.traffic_type