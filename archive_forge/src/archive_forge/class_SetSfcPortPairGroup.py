import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
class SetSfcPortPairGroup(command.Command):
    _description = _('Set port pair group properties')

    def get_parser(self, prog_name):
        parser = super(SetSfcPortPairGroup, self).get_parser(prog_name)
        parser.add_argument('port_pair_group', metavar='<port-pair-group>', help=_('Port pair group to modify (name or ID)'))
        parser.add_argument('--name', metavar='<name>', help=_('Name of the port pair group'))
        parser.add_argument('--description', metavar='<description>', help=_('Description for the port pair group'))
        parser.add_argument('--port-pair', metavar='<port-pair>', dest='port_pairs', default=[], action='append', help=_('Port pair (name or ID). This option can be repeated.'))
        parser.add_argument('--no-port-pair', action='store_true', help=_('Remove all port pair from port pair group'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        ppg_id = client.find_sfc_port_pair_group(parsed_args.port_pair_group)['id']
        attrs = _get_common_attrs(self.app.client_manager, parsed_args, is_create=False)
        if parsed_args.no_port_pair:
            attrs['port_pairs'] = []
        if parsed_args.port_pairs:
            added = [client.find_sfc_port_pair(pp, ignore_missing=False)['id'] for pp in parsed_args.port_pairs]
            if parsed_args.no_port_pair:
                existing = []
            else:
                existing = client.find_sfc_port_pair_group(parsed_args.port_pair_group, ignore_missing=False)['port_pairs']
            attrs['port_pairs'] = sorted(list(set(existing) | set(added)))
        try:
            client.update_sfc_port_pair_group(ppg_id, **attrs)
        except Exception as e:
            msg = _("Failed to update port pair group '%(ppg)s': %(e)s") % {'ppg': parsed_args.port_pair_group, 'e': e}
            raise exceptions.CommandError(msg)