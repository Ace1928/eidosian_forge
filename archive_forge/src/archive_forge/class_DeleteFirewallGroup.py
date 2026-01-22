import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import constants as const
from neutronclient.osc.v2 import utils as v2_utils
class DeleteFirewallGroup(command.Command):
    _description = _('Delete firewall group(s)')

    def get_parser(self, prog_name):
        parser = super(DeleteFirewallGroup, self).get_parser(prog_name)
        parser.add_argument(const.FWG, metavar='<firewall-group>', nargs='+', help=_('Firewall group(s) to delete (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        result = 0
        for fwg in parsed_args.firewall_group:
            try:
                fwg_id = client.find_firewall_group(fwg)['id']
                client.delete_firewall_group(fwg_id)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete firewall group with name or ID '%(firewall_group)s': %(e)s"), {const.FWG: fwg, 'e': e})
        if result > 0:
            total = len(parsed_args.firewall_group)
            msg = _('%(result)s of %(total)s firewall group(s) failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)