import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
class UnsetSfcPortChain(command.Command):
    _description = _('Unset port chain properties')

    def get_parser(self, prog_name):
        parser = super(UnsetSfcPortChain, self).get_parser(prog_name)
        parser.add_argument('port_chain', metavar='<port-chain>', help=_('Port chain to unset (name or ID)'))
        port_chain = parser.add_mutually_exclusive_group()
        port_chain.add_argument('--flow-classifier', action='append', metavar='<flow-classifier>', dest='flow_classifiers', help=_('Remove flow classifier(s) from the port chain (name or ID). This option can be repeated.'))
        port_chain.add_argument('--all-flow-classifier', action='store_true', help=_('Remove all flow classifiers from the port chain'))
        parser.add_argument('--port-pair-group', metavar='<port-pair-group>', dest='port_pair_groups', action='append', help=_('Remove port pair group(s) from the port chain (name or ID). This option can be repeated.'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        pc_id = client.find_sfc_port_chain(parsed_args.port_chain, ignore_missing=False)['id']
        attrs = {}
        if parsed_args.flow_classifiers:
            fc_list = client.find_sfc_port_chain(parsed_args.port_chain, ignore_missing=False)['flow_classifiers']
            for fc in parsed_args.flow_classifiers:
                fc_id = client.find_sfc_flow_classifier(fc, ignore_missing=False)['id']
                if fc_id in fc_list:
                    fc_list.remove(fc_id)
            attrs['flow_classifiers'] = fc_list
        if parsed_args.all_flow_classifier:
            attrs['flow_classifiers'] = []
        if parsed_args.port_pair_groups:
            ppg_list = client.find_sfc_port_chain(parsed_args.port_chain, ignore_missing=False)['port_pair_groups']
            for ppg in parsed_args.port_pair_groups:
                ppg_id = client.find_sfc_port_pair_group(ppg, ignore_missing=False)['id']
                if ppg_id in ppg_list:
                    ppg_list.remove(ppg_id)
            if ppg_list == []:
                message = _('At least one port pair group must be specified.')
                raise exceptions.CommandError(message)
            attrs['port_pair_groups'] = ppg_list
        try:
            client.update_sfc_port_chain(pc_id, **attrs)
        except Exception as e:
            msg = _("Failed to unset port chain '%(pc)s': %(e)s") % {'pc': parsed_args.port_chain, 'e': e}
            raise exceptions.CommandError(msg)