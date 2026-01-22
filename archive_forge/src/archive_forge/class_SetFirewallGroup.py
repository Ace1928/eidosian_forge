import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import constants as const
from neutronclient.osc.v2 import utils as v2_utils
class SetFirewallGroup(command.Command):
    _description = _('Set firewall group properties')

    def get_parser(self, prog_name):
        parser = super(SetFirewallGroup, self).get_parser(prog_name)
        _get_common_parser(parser)
        parser.add_argument(const.FWG, metavar='<firewall-group>', help=_('Firewall group to update (name or ID)'))
        parser.add_argument('--port', metavar='<port>', action='append', help=_('Port(s) (name or ID) to apply firewall group.  This option can be repeated'))
        parser.add_argument('--no-port', dest='no_port', action='store_true', help=_('Detach all port from the firewall group'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        fwg_id = client.find_firewall_group(parsed_args.firewall_group)['id']
        attrs = _get_common_attrs(self.app.client_manager, parsed_args, is_create=False)
        try:
            client.update_firewall_group(fwg_id, **attrs)
        except Exception as e:
            msg = _("Failed to set firewall group '%(group)s': %(e)s") % {'group': parsed_args.firewall_group, 'e': e}
            raise exceptions.CommandError(msg)