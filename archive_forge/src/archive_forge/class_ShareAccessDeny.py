import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.osc import utils
class ShareAccessDeny(command.Command):
    """Delete a share access rule."""
    _description = _('Delete a share access rule')

    def get_parser(self, prog_name):
        parser = super(ShareAccessDeny, self).get_parser(prog_name)
        parser.add_argument('share', metavar='<share>', help=_('Name or ID of the NAS share to modify.'))
        parser.add_argument('id', metavar='<id>', help=_('ID of the access rule to be deleted.'))
        parser.add_argument('--wait', action='store_true', default=False, help=_('Wait for share access rule deletion'))
        parser.add_argument('--unrestrict', action='store_true', default=False, help=_('Seek access rule deletion despite restrictions. Only available with API version >= 2.82.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share = apiutils.find_resource(share_client.shares, parsed_args.share)
        kwargs = {}
        if parsed_args.unrestrict:
            if share_client.api_version < api_versions.APIVersion('2.82'):
                raise exceptions.CommandError('Restricted access rules are only available starting from API version 2.82.')
            kwargs['unrestrict'] = True
        error = None
        try:
            share.deny(parsed_args.id, **kwargs)
            if parsed_args.wait:
                if not oscutils.wait_for_delete(manager=share_client.share_access_rules, res_id=parsed_args.id):
                    error = _('Failed to delete share access rule with ID: %s' % parsed_args.id)
        except Exception as e:
            error = e
        if error:
            raise exceptions.CommandError(_("Failed to delete share access rule for share '%s': %s" % (share, error)))