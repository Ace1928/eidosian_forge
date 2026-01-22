import argparse
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0 import dns
from neutronclient.neutron.v2_0.qos import policy as qos_policy
def args2body_allowedaddresspairs(self, parsed_args, port):
    if parsed_args.allowed_address_pairs:
        port['allowed_address_pairs'] = parsed_args.allowed_address_pairs
    elif parsed_args.no_allowed_address_pairs:
        port['allowed_address_pairs'] = []