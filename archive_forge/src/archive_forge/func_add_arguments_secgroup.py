import argparse
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0 import dns
from neutronclient.neutron.v2_0.qos import policy as qos_policy
def add_arguments_secgroup(self, parser):
    group_sg = parser.add_mutually_exclusive_group()
    group_sg.add_argument('--security-group', metavar='SECURITY_GROUP', default=[], action='append', dest='security_groups', help=_('Security group associated with the port. You can repeat this option.'))
    group_sg.add_argument('--no-security-groups', action='store_true', help=_('Associate no security groups with the port.'))