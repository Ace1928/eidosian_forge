import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ShowRegion(command.ShowOne):
    _description = _('Display region details')

    def get_parser(self, prog_name):
        parser = super(ShowRegion, self).get_parser(prog_name)
        parser.add_argument('region', metavar='<region-id>', help=_('Region to display'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        region = utils.find_resource(identity_client.regions, parsed_args.region)
        region._info['region'] = region._info.pop('id')
        region._info['parent_region'] = region._info.pop('parent_region_id')
        region._info.pop('links', None)
        return zip(*sorted(region._info.items()))