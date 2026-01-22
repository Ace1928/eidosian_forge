import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common._i18n import _
class DeleteShareSecurityService(command.Command):
    """Delete one or more security services."""
    _description = _('Delete one or more security services.')

    def get_parser(self, prog_name):
        parser = super(DeleteShareSecurityService, self).get_parser(prog_name)
        parser.add_argument('security_service', metavar='<security-service>', nargs='+', help=_('Name or ID of the security service(s) to delete.'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        result = 0
        for security_service in parsed_args.security_service:
            try:
                security_service_obj = oscutils.find_resource(share_client.security_services, security_service)
                share_client.security_services.delete(security_service_obj)
            except Exception as e:
                result += 1
                LOG.error(f'Failed to delete security service with name or ID {security_service}: {e}')
        if result > 0:
            total = len(parsed_args.security_service)
            msg = f'{result} of {total} security services failed to be deleted.'
            raise exceptions.CommandError(msg)