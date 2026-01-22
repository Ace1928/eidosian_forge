import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common._i18n import _
class CreateShareSecurityService(command.ShowOne):
    """Create security service used by project."""
    _description = _('Create security service used by project.')

    def get_parser(self, prog_name):
        parser = super(CreateShareSecurityService, self).get_parser(prog_name)
        parser.add_argument('type', metavar='<type>', default=None, choices=['ldap', 'kerberos', 'active_directory'], help=_("Security service type. Possible options are: 'ldap', 'kerberos', 'active_directory'."))
        parser.add_argument('--dns-ip', metavar='<dns-ip>', default=None, help=_("DNS IP address of the security service used inside project's network."))
        parser.add_argument('--ou', metavar='<ou>', default=None, help=_('Security service OU (Organizational Unit). Available only for microversion >= 2.44.'))
        parser.add_argument('--server', metavar='<server>', default=None, help=_('Security service IP address or hostname.'))
        parser.add_argument('--domain', metavar='<domain>', default=None, help=_('Security service domain.'))
        parser.add_argument('--user', metavar='<user', default=None, help=_('Security service user or group used by project.'))
        parser.add_argument('--password', metavar='<password>', default=None, help=_('Password used by user.'))
        parser.add_argument('--name', metavar='<name>', default=None, help=_('Security service name.'))
        parser.add_argument('--description', metavar='<description>', default=None, help=_('Security service description.'))
        parser.add_argument('--default-ad-site', metavar='<default_ad_site>', dest='default_ad_site', default=None, help=_("Default AD site. Available only for microversion >= 2.76. Can be provided in the place of '--server' but not along with it."))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        kwargs = {'dns_ip': parsed_args.dns_ip, 'server': parsed_args.server, 'domain': parsed_args.domain, 'user': parsed_args.user, 'password': parsed_args.password, 'name': parsed_args.name, 'description': parsed_args.description}
        if share_client.api_version >= api_versions.APIVersion('2.44'):
            kwargs['ou'] = parsed_args.ou
        elif parsed_args.ou:
            raise exceptions.CommandError('Defining a security service Organizational Unit is available only for microversion >= 2.44')
        if share_client.api_version >= api_versions.APIVersion('2.76'):
            kwargs['default_ad_site'] = parsed_args.default_ad_site
        elif parsed_args.default_ad_site:
            raise exceptions.CommandError('Defining a security service Default AD site is available only for microversion >= 2.76')
        if parsed_args.type == 'active_directory':
            server = parsed_args.server
            default_ad_site = parsed_args.default_ad_site
            if server and default_ad_site:
                raise exceptions.CommandError("Cannot create security service because both server and 'default_ad_site' were provided. Specify either server or 'default_ad_site'.")
        security_service = share_client.security_services.create(parsed_args.type, **kwargs)
        return self.dict2columns(security_service._info)