import itertools
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class SetNetworkSegmentRange(common.NeutronCommandWithExtraArgs):
    _description = _('Set network segment range properties')

    def get_parser(self, prog_name):
        parser = super(SetNetworkSegmentRange, self).get_parser(prog_name)
        parser.add_argument('network_segment_range', metavar='<network-segment-range>', help=_('Network segment range to modify (name or ID)'))
        parser.add_argument('--name', metavar='<name>', help=_('Set network segment name'))
        parser.add_argument('--minimum', metavar='<minimum-segmentation-id>', type=int, help=_('Set network segment range minimum segment identifier'))
        parser.add_argument('--maximum', metavar='<maximum-segmentation-id>', type=int, help=_('Set network segment range maximum segment identifier'))
        return parser

    def take_action(self, parsed_args):
        network_client = self.app.client_manager.network
        try:
            network_client.find_extension('network-segment-range', ignore_missing=False)
        except Exception as e:
            msg = _('Network segment range set not supported by Network API: %(e)s') % {'e': e}
            raise exceptions.CommandError(msg)
        if parsed_args.minimum and (not parsed_args.maximum) or (parsed_args.maximum and (not parsed_args.minimum)):
            msg = _('--minimum and --maximum are both required')
            raise exceptions.CommandError(msg)
        obj = network_client.find_network_segment_range(parsed_args.network_segment_range, ignore_missing=False)
        attrs = {}
        if parsed_args.name:
            attrs['name'] = parsed_args.name
        if parsed_args.minimum:
            attrs['minimum'] = parsed_args.minimum
        if parsed_args.maximum:
            attrs['maximum'] = parsed_args.maximum
        attrs.update(self._parse_extra_properties(parsed_args.extra_properties))
        network_client.update_network_segment_range(obj, **attrs)