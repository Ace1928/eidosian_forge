import logging
import netaddr
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
def _format_addresses(addresses):
    return [str(netaddr.IPNetwork(addr)) for addr in addresses]