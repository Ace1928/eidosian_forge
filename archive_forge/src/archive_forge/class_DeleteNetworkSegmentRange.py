import itertools
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class DeleteNetworkSegmentRange(command.Command):
    _description = _('Delete network segment range(s)')

    def get_parser(self, prog_name):
        parser = super(DeleteNetworkSegmentRange, self).get_parser(prog_name)
        parser.add_argument('network_segment_range', metavar='<network-segment-range>', nargs='+', help=_('Network segment range(s) to delete (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        network_client = self.app.client_manager.network
        try:
            network_client.find_extension('network-segment-range', ignore_missing=False)
        except Exception as e:
            msg = _('Network segment range delete not supported by Network API: %(e)s') % {'e': e}
            raise exceptions.CommandError(msg)
        result = 0
        for network_segment_range in parsed_args.network_segment_range:
            try:
                obj = network_client.find_network_segment_range(network_segment_range, ignore_missing=False)
                network_client.delete_network_segment_range(obj)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete network segment range with ID '%(network_segment_range)s': %(e)s"), {'network_segment_range': network_segment_range, 'e': e})
        if result > 0:
            total = len(parsed_args.network_segment_range)
            msg = _('%(result)s of %(total)s network segment ranges failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)