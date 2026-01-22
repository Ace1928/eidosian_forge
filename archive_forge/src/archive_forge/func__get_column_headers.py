import argparse
import logging
from osc_lib.cli import parseractions
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
from openstackclient.network import utils as network_utils
def _get_column_headers(self, parsed_args):
    column_headers = ('ID', 'IP Protocol', 'Ethertype', 'IP Range', 'Port Range', 'Direction', 'Remote Security Group')
    if self.is_neutron:
        column_headers = column_headers + ('Remote Address Group',)
    if parsed_args.group is None:
        column_headers = column_headers + ('Security Group',)
    return column_headers