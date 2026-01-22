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
class ResizeShare(command.Command):
    """Resize a share"""
    _description = _('Resize a share')

    def get_parser(self, prog_name):
        parser = super(ResizeShare, self).get_parser(prog_name)
        parser.add_argument('share', metavar='<share>', help=_('Name or ID of share to resize'))
        parser.add_argument('new_size', metavar='<new-size>', type=int, help=_('New size of share, in GiBs'))
        parser.add_argument('--wait', action='store_true', default=False, help=_('Wait for share resize'))
        parser.add_argument('--force', action='store_true', default=False, help=_('Only applicable when increasing the size of the share，only available with microversion 2.64 and higher. (admin only)'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share = apiutils.find_resource(share_client.shares, parsed_args.share)
        share_size = share._info['size']
        new_size = parsed_args.new_size
        if share_size > new_size:
            try:
                share_client.shares.shrink(share, new_size)
            except Exception as e:
                raise exceptions.CommandError(_('Share resize failed: %s' % e))
        elif share_size < new_size:
            force = False
            if parsed_args.force:
                if share_client.api_version < api_versions.APIVersion('2.64'):
                    raise exceptions.CommandError('args force is available only for API microversion >= 2.64')
                force = True
            try:
                if force:
                    share_client.shares.extend(share, new_size, force=force)
                else:
                    share_client.shares.extend(share, new_size)
            except Exception as e:
                raise exceptions.CommandError(_('Share resize failed: %s' % e))
        else:
            raise exceptions.CommandError(_('Share size is already at %s GiBs' % new_size))
        if parsed_args.wait:
            if not oscutils.wait_for_status(status_f=share_client.shares.get, res_id=share.id, success_status=['available']):
                raise exceptions.CommandError(_('Share not available after resize attempt.'))