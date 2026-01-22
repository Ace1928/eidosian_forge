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
def _convert_address_pairs(parsed_args):
    ops = []
    for opt in parsed_args.allowed_address_pairs:
        addr = {}
        addr['ip_address'] = opt['ip-address']
        if 'mac-address' in opt:
            addr['mac_address'] = opt['mac-address']
        ops.append(addr)
    return ops