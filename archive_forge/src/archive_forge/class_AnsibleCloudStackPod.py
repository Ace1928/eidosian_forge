from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
class AnsibleCloudStackPod(AnsibleCloudStack):

    def __init__(self, module):
        super(AnsibleCloudStackPod, self).__init__(module)
        self.returns = {'endip': 'end_ip', 'startip': 'start_ip', 'gateway': 'gateway', 'netmask': 'netmask', 'allocationstate': 'allocation_state'}
        self.pod = None

    def _get_common_pod_args(self):
        args = {'name': self.module.params.get('name'), 'zoneid': self.get_zone(key='id'), 'startip': self.module.params.get('start_ip'), 'endip': self.module.params.get('end_ip'), 'netmask': self.module.params.get('netmask'), 'gateway': self.module.params.get('gateway')}
        state = self.module.params.get('state')
        if state in ['enabled', 'disabled']:
            args['allocationstate'] = state.capitalize()
        return args

    def get_pod(self):
        if not self.pod:
            args = {'zoneid': self.get_zone(key='id')}
            uuid = self.module.params.get('id')
            if uuid:
                args['id'] = uuid
            else:
                args['name'] = self.module.params.get('name')
            pods = self.query_api('listPods', **args)
            if pods:
                for pod in pods['pod']:
                    if not args['name']:
                        self.pod = self._transform_ip_list(pod)
                        break
                    elif args['name'] == pod['name']:
                        self.pod = self._transform_ip_list(pod)
                        break
        return self.pod

    def present_pod(self):
        pod = self.get_pod()
        if pod:
            pod = self._update_pod()
        else:
            pod = self._create_pod()
        return pod

    def _create_pod(self):
        required_params = ['start_ip', 'netmask', 'gateway']
        self.module.fail_on_missing_params(required_params=required_params)
        pod = None
        self.result['changed'] = True
        args = self._get_common_pod_args()
        if not self.module.check_mode:
            res = self.query_api('createPod', **args)
            pod = res['pod']
        return pod

    def _update_pod(self):
        pod = self.get_pod()
        args = self._get_common_pod_args()
        args['id'] = pod['id']
        if self.has_changed(args, pod):
            self.result['changed'] = True
            if not self.module.check_mode:
                res = self.query_api('updatePod', **args)
                pod = res['pod']
        return pod

    def absent_pod(self):
        pod = self.get_pod()
        if pod:
            self.result['changed'] = True
            args = {'id': pod['id']}
            if not self.module.check_mode:
                self.query_api('deletePod', **args)
        return pod

    def _transform_ip_list(self, resource):
        """ Workaround for 4.11 return API break """
        keys = ['endip', 'startip']
        if resource:
            for key in keys:
                if key in resource and isinstance(resource[key], list):
                    resource[key] = resource[key][0]
        return resource

    def get_result(self, resource):
        resource = self._transform_ip_list(resource)
        super(AnsibleCloudStackPod, self).get_result(resource)
        return self.result