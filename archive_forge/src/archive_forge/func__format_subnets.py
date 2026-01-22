import argparse
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0 import availability_zone
from neutronclient.neutron.v2_0 import dns
from neutronclient.neutron.v2_0.qos import policy as qos_policy
def _format_subnets(network):
    try:
        return '\n'.join([' '.join([s['id'], s.get('cidr', '')]) for s in network['subnets']])
    except (TypeError, KeyError):
        return ''