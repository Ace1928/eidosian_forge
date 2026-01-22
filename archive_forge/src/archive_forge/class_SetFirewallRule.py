import logging
from cliff import columns as cliff_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.common import utils as nc_utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import constants as const
class SetFirewallRule(command.Command):
    _description = _('Set firewall rule properties')

    def get_parser(self, prog_name):
        parser = super(SetFirewallRule, self).get_parser(prog_name)
        _get_common_parser(parser)
        parser.add_argument(const.FWR, metavar='<firewall-rule>', help=_('Firewall rule to set (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        attrs = _get_common_attrs(self.app.client_manager, parsed_args, is_create=False)
        fwr_id = client.find_firewall_rule(parsed_args.firewall_rule)['id']
        try:
            client.update_firewall_rule(fwr_id, **attrs)
        except Exception as e:
            msg = _("Failed to set firewall rule '%(rule)s': %(e)s") % {'rule': parsed_args.firewall_rule, 'e': e}
            raise exceptions.CommandError(msg)