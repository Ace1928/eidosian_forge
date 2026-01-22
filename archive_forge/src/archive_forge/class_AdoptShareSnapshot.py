import logging
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common import cliutils
from manilaclient.osc import utils as oscutils
class AdoptShareSnapshot(command.ShowOne):
    """Adopt a share snapshot not handled by Manila (Admin only)."""
    _description = _('Adopt a share snapshot')

    def get_parser(self, prog_name):
        parser = super(AdoptShareSnapshot, self).get_parser(prog_name)
        parser.add_argument('share', metavar='<share>', help=_('Name or ID of the share that owns the snapshot to be adopted.'))
        parser.add_argument('provider_location', metavar='<provider-location>', help=_('Provider location of the snapshot on the backend.'))
        parser.add_argument('--name', metavar='<name>', default=None, help=_('Optional snapshot name (Default=None).'))
        parser.add_argument('--description', metavar='<description>', default=None, help=_('Optional snapshot description (Default=None).'))
        parser.add_argument('--driver-option', metavar='<key=value>', default={}, action=parseractions.KeyValueAction, help=_('Set driver options as key=value pairs.(repeat option to set multiple key=value pairs)'))
        parser.add_argument('--wait', action='store_true', help=_('Wait until share snapshot is adopted'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share = utils.find_resource(share_client.shares, parsed_args.share)
        snapshot = share_client.share_snapshots.manage(share=share, provider_location=parsed_args.provider_location, driver_options=parsed_args.driver_option, name=parsed_args.name, description=parsed_args.description)
        if parsed_args.wait:
            if not utils.wait_for_status(status_f=share_client.share_snapshots.get, res_id=snapshot.id, success_status=['available'], error_status=['manage_error', 'error']):
                LOG.error(_('ERROR: Share snapshot is in error state.'))
            snapshot = utils.find_resource(share_client.share_snapshots, snapshot.id)
        snapshot._info.pop('links', None)
        return self.dict2columns(snapshot._info)