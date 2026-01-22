import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class SetRegion(command.Command):
    _description = _('Set region properties')

    def get_parser(self, prog_name):
        parser = super(SetRegion, self).get_parser(prog_name)
        parser.add_argument('region', metavar='<region-id>', help=_('Region to modify'))
        parser.add_argument('--parent-region', metavar='<region-id>', help=_('New parent region ID'))
        parser.add_argument('--description', metavar='<description>', help=_('New region description'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        kwargs = {}
        if parsed_args.description:
            kwargs['description'] = parsed_args.description
        if parsed_args.parent_region:
            kwargs['parent_region'] = parsed_args.parent_region
        identity_client.regions.update(parsed_args.region, **kwargs)