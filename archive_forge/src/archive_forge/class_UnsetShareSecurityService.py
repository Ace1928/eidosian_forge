import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common._i18n import _
class UnsetShareSecurityService(command.Command):
    """Unset security service."""
    _description = _('Unset security service.')

    def get_parser(self, prog_name):
        parser = super(UnsetShareSecurityService, self).get_parser(prog_name)
        parser.add_argument('security_service', metavar='<security-service>', help=_('Security service name or ID.'))
        parser.add_argument('--dns-ip', action='store_true', help=_("Unset DNS IP address used inside project's network."))
        parser.add_argument('--ou', action='store_true', help=_('Unset security service OU (Organizational Unit). Available only for microversion >= 2.44.'))
        parser.add_argument('--server', action='store_true', help=_('Unset security service IP address or hostname.'))
        parser.add_argument('--domain', action='store_true', help=_('Unset security service domain.'))
        parser.add_argument('--user', action='store_true', help=_('Unset security service user or group used by project.'))
        parser.add_argument('--password', action='store_true', help=_('Unset password used by user.'))
        parser.add_argument('--name', action='store_true', help=_('Unset security service name.'))
        parser.add_argument('--description', action='store_true', help=_('Unset security service description.'))
        parser.add_argument('--default-ad-site', dest='default_ad_site', action='store_true', help=_('Default AD site. Available only for microversion >= 2.76.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        security_service = oscutils.find_resource(share_client.security_services, parsed_args.security_service)
        kwargs = {}
        args = ['dns_ip', 'server', 'domain', 'user', 'password', 'name', 'description']
        for arg in args:
            if getattr(parsed_args, arg):
                kwargs[arg] = ''
        if parsed_args.ou and share_client.api_version >= api_versions.APIVersion('2.44'):
            kwargs['ou'] = ''
        elif parsed_args.ou:
            raise exceptions.CommandError(_('Unsetting a security service Organizational Unit is available only for microversion >= 2.44'))
        if parsed_args.default_ad_site and share_client.api_version >= api_versions.APIVersion('2.76'):
            kwargs['default_ad_site'] = ''
        elif parsed_args.default_ad_site:
            raise exceptions.CommandError(_('Unsetting a security service Default AD site is available only for microversion >= 2.76'))
        try:
            security_service.update(**kwargs)
        except Exception as e:
            raise exceptions.CommandError(f'One or more unset operations failed: {e}')