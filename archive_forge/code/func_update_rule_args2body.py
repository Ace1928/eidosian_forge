from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.qos import policy as qos_policy
def update_rule_args2body(parsed_args, body):
    neutronv20.update_dict(parsed_args, body, ['rule'])