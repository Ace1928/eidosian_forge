import json
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class DeleteMapping(command.Command):
    _description = _('Delete mapping(s)')

    def get_parser(self, prog_name):
        parser = super(DeleteMapping, self).get_parser(prog_name)
        parser.add_argument('mapping', metavar='<mapping>', nargs='+', help=_('Mapping(s) to delete'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        result = 0
        for i in parsed_args.mapping:
            try:
                identity_client.federation.mappings.delete(i)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete mapping with name or ID '%(mapping)s': %(e)s"), {'mapping': i, 'e': e})
        if result > 0:
            total = len(parsed_args.mapping)
            msg = _('%(result)s of %(total)s mappings failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)