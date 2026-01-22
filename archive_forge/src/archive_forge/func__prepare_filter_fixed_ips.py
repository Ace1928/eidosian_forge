import argparse
import copy
import json
import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import tags as _tag
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
def _prepare_filter_fixed_ips(client_manager, parsed_args):
    """Fix and properly format fixed_ip option for filtering.

    Appropriately convert any subnet names to their respective ids.
    Convert fixed_ips in parsed args to be in valid list format for filter:
    ['subnet_id=foo'].
    """
    client = client_manager.network
    ips = []
    for ip_spec in parsed_args.fixed_ip:
        if 'subnet' in ip_spec:
            subnet_name_id = ip_spec['subnet']
            if subnet_name_id:
                _subnet = client.find_subnet(subnet_name_id, ignore_missing=False)
                ips.append('subnet_id=%s' % _subnet.id)
        if 'ip-address' in ip_spec:
            ips.append('ip_address=%s' % ip_spec['ip-address'])
        if 'ip-substring' in ip_spec:
            ips.append('ip_address_substr=%s' % ip_spec['ip-substring'])
    return ips