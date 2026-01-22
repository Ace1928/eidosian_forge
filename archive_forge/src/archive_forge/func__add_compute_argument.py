import argparse
from unittest import mock
import openstack
from osc_lib import exceptions
from openstackclient.network import common
from openstackclient.tests.unit import utils
def _add_compute_argument(parser):
    parser.add_argument('compute', metavar='<compute>', help='Compute argument')
    return parser