import os
from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronv20
def args2body_qos_policy(self, parsed_args, resource):
    super(UpdateQosPolicyMixin, self).args2body_qos_policy(parsed_args, resource)
    if parsed_args.no_qos_policy:
        resource['qos_policy_id'] = None