from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
from neutronclient.neutron.v2_0.qos import rule as qos_rule
def update_minimum_bandwidth_args2body(parsed_args, body):
    neutronv20.update_dict(parsed_args, body, ['min_kbps', 'direction'])