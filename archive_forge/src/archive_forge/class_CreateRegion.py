import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class CreateRegion(command.ShowOne):
    _description = _('Create new region')

    def get_parser(self, prog_name):
        parser = super(CreateRegion, self).get_parser(prog_name)
        parser.add_argument('region', metavar='<region-id>', help=_('New region ID'))
        parser.add_argument('--parent-region', metavar='<region-id>', help=_('Parent region ID'))
        parser.add_argument('--description', metavar='<description>', help=_('New region description'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        region = identity_client.regions.create(id=parsed_args.region, parent_region=parsed_args.parent_region, description=parsed_args.description)
        region._info['region'] = region._info.pop('id')
        region._info['parent_region'] = region._info.pop('parent_region_id')
        region._info.pop('links', None)
        return zip(*sorted(region._info.items()))