from __future__ import absolute_import, division, print_function
from ..module_utils.cloudstack import AnsibleCloudStack, cs_argument_spec, cs_required_together
from ansible.module_utils.basic import AnsibleModule
class AnsibleCloudStackTrafficType(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackTrafficType, self).__init__(module)
        self.returns = {'traffictype': 'traffic_type', 'hypervnetworklabel': 'hyperv_networklabel', 'kvmnetworklabel': 'kvm_networklabel', 'ovm3networklabel': 'ovm3_networklabel', 'physicalnetworkid': 'physical_network', 'vmwarenetworklabel': 'vmware_networklabel', 'xennetworklabel': 'xen_networklabel'}
        self.traffic_type = None

    def _get_label_args(self):
        label_args = dict()
        if self.module.params.get('hyperv_networklabel'):
            label_args.update(dict(hypervnetworklabel=self.module.params.get('hyperv_networklabel')))
        if self.module.params.get('kvm_networklabel'):
            label_args.update(dict(kvmnetworklabel=self.module.params.get('kvm_networklabel')))
        if self.module.params.get('ovm3_networklabel'):
            label_args.update(dict(ovm3networklabel=self.module.params.get('ovm3_networklabel')))
        if self.module.params.get('vmware_networklabel'):
            label_args.update(dict(vmwarenetworklabel=self.module.params.get('vmware_networklabel')))
        return label_args

    def _get_additional_args(self):
        additional_args = dict()
        if self.module.params.get('isolation_method'):
            additional_args.update(dict(isolationmethod=self.module.params.get('isolation_method')))
        if self.module.params.get('vlan'):
            additional_args.update(dict(vlan=self.module.params.get('vlan')))
        additional_args.update(self._get_label_args())
        return additional_args

    def get_traffic_types(self):
        args = {'physicalnetworkid': self.get_physical_network(key='id')}
        traffic_types = self.query_api('listTrafficTypes', **args)
        return traffic_types

    def get_traffic_type(self):
        if self.traffic_type:
            return self.traffic_type
        traffic_type = self.module.params.get('traffic_type')
        traffic_types = self.get_traffic_types()
        if traffic_types:
            for t_type in traffic_types['traffictype']:
                if traffic_type.lower() in [t_type['traffictype'].lower(), t_type['id']]:
                    self.traffic_type = t_type
                    break
        return self.traffic_type

    def present_traffic_type(self):
        traffic_type = self.get_traffic_type()
        if traffic_type:
            self.traffic_type = self.update_traffic_type()
        else:
            self.result['changed'] = True
            self.traffic_type = self.add_traffic_type()
        return self.traffic_type

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

    def absent_traffic_type(self):
        traffic_type = self.get_traffic_type()
        if traffic_type:
            args = {'id': traffic_type['id']}
            self.result['changed'] = True
            if not self.module.check_mode:
                resource = self.query_api('deleteTrafficType', **args)
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    self.poll_job(resource, 'traffictype')
        return traffic_type

    def update_traffic_type(self):
        traffic_type = self.get_traffic_type()
        args = {'id': traffic_type['id']}
        args.update(self._get_label_args())
        if self.has_changed(args, traffic_type):
            self.result['changed'] = True
            if not self.module.check_mode:
                resource = self.query_api('updateTrafficType', **args)
                poll_async = self.module.params.get('poll_async')
                if poll_async:
                    self.traffic_type = self.poll_job(resource, 'traffictype')
        return self.traffic_type