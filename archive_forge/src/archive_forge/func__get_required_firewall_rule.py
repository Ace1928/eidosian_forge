import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import constants as const
def _get_required_firewall_rule(client, parsed_args):
    if not parsed_args.firewall_rule:
        msg = _('Firewall rule (name or ID) is required.')
        raise exceptions.CommandError(msg)
    return client.find_firewall_rule(parsed_args.firewall_rule)['id']