import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
class UnsetSfcPortPairGroup(command.Command):
    _description = _('Unset port pairs from port pair group')

    def get_parser(self, prog_name):
        parser = super(UnsetSfcPortPairGroup, self).get_parser(prog_name)
        parser.add_argument('port_pair_group', metavar='<port-pair-group>', help=_('Port pair group to unset (name or ID)'))
        port_pair_group = parser.add_mutually_exclusive_group()
        port_pair_group.add_argument('--port-pair', action='append', metavar='<port-pair>', dest='port_pairs', help=_('Remove port pair(s) from the port pair group (name or ID). This option can be repeated.'))
        port_pair_group.add_argument('--all-port-pair', action='store_true', help=_('Remove all port pairs from the port pair group'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        ppg_id = client.find_sfc_port_pair_group(parsed_args.port_pair_group, ignore_missing=False)['id']
        attrs = {}
        if parsed_args.port_pairs:
            existing = client.find_sfc_port_pair_group(parsed_args.port_pair_group, ignore_missing=False)['port_pairs']
            removed = [client.find_sfc_port_pair(pp, ignore_missing=False)['id'] for pp in parsed_args.port_pairs]
            attrs['port_pairs'] = list(set(existing) - set(removed))
        if parsed_args.all_port_pair:
            attrs['port_pairs'] = []
        try:
            client.update_sfc_port_pair_group(ppg_id, **attrs)
        except Exception as e:
            msg = _("Failed to unset port pair group '%(ppg)s': %(e)s") % {'ppg': parsed_args.port_pair_group, 'e': e}
            raise exceptions.CommandError(msg)