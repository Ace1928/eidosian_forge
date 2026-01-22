import logging
from openstackclient.identity import common as identity_common
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import exceptions as apiclient_exceptions
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.common import cliutils
from manilaclient.osc import utils
class ShareExportLocationList(command.Lister):
    """List export locations of a share."""
    _description = _('List export location of a share')

    def get_parser(self, prog_name):
        parser = super(ShareExportLocationList, self).get_parser(prog_name)
        parser.add_argument('share', metavar='<share>', help=_('Name or ID of share'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share = apiutils.find_resource(share_client.shares, parsed_args.share)
        export_locations = share_client.share_export_locations.list(share=share)
        list_of_keys = ['ID', 'Path', 'Preferred']
        return (list_of_keys, (oscutils.get_item_properties(s, list_of_keys) for s in export_locations))