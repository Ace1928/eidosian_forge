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
class ShowShare(command.ShowOne):
    """Show a share."""
    _description = _('Display share details')

    def get_parser(self, prog_name):
        parser = super(ShowShare, self).get_parser(prog_name)
        parser.add_argument('share', metavar='<share>', help=_('Share to display (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share_obj = apiutils.find_resource(share_client.shares, parsed_args.share)
        export_locations = share_client.share_export_locations.list(share_obj)
        export_locations = cliutils.convert_dict_list_to_string(export_locations, ignored_keys=['replica_state', 'availability_zone', 'share_replica_id'])
        data = share_obj._info
        data['export_locations'] = export_locations
        data.update({'properties': format_columns.DictColumn(data.pop('metadata', {}))})
        data.pop('links', None)
        data.pop('shares_type', None)
        return self.dict2columns(data)