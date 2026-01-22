import os
from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronv20
class CreateQosPolicyMixin(object):

    def add_arguments_qos_policy(self, parser):
        qos_policy_args = parser.add_mutually_exclusive_group()
        qos_policy_args.add_argument('--qos-policy', help=_('ID or name of the QoS policy that shouldbe attached to the resource.'))
        return qos_policy_args

    def args2body_qos_policy(self, parsed_args, resource):
        if parsed_args.qos_policy:
            _policy_id = get_qos_policy_id(self.get_client(), parsed_args.qos_policy)
            resource['qos_policy_id'] = _policy_id