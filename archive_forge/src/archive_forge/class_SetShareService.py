from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from manilaclient import api_versions
from manilaclient.common._i18n import _
class SetShareService(command.Command):
    """Enable/disable share service (Admin only)."""
    _description = _('Enable/Disable share service (Admin only).')

    def get_parser(self, prog_name):
        parser = super(SetShareService, self).get_parser(prog_name)
        parser.add_argument('host', metavar='<host>', help=_("Host name as 'example_host@example_backend'."))
        parser.add_argument('binary', metavar='<binary>', help=_("Service binary, could be 'manila-share', 'manila-scheduler' or 'manila-data'"))
        enable_group = parser.add_mutually_exclusive_group()
        enable_group.add_argument('--enable', action='store_true', help=_('Enable share service'))
        enable_group.add_argument('--disable', action='store_true', help=_('Disable share service'))
        parser.add_argument('--disable-reason', metavar='<reason>', help=_('Reason for disabling the service (should be used with --disable option)'))
        return parser

    def take_action(self, parsed_args):
        if parsed_args.disable_reason and (not parsed_args.disable):
            msg = _('Cannot specify option --disable-reason without --disable specified.')
            raise exceptions.CommandError(msg)
        share_client = self.app.client_manager.share
        if parsed_args.enable:
            try:
                share_client.services.enable(parsed_args.host, parsed_args.binary)
            except Exception as e:
                raise exceptions.CommandError(_('Failed to enable service: %s' % e))
        if parsed_args.disable:
            if parsed_args.disable_reason:
                if share_client.api_version < api_versions.APIVersion('2.83'):
                    raise exceptions.CommandError('Service disable reason can be specified only with manila API version >= 2.83')
            try:
                if parsed_args.disable_reason:
                    share_client.services.disable(parsed_args.host, parsed_args.binary, disable_reason=parsed_args.disable_reason)
                else:
                    share_client.services.disable(parsed_args.host, parsed_args.binary)
            except Exception as e:
                raise exceptions.CommandError(_('Failed to disable service: %s' % e))