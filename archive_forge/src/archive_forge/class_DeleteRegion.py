import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class DeleteRegion(command.Command):
    _description = _('Delete region(s)')

    def get_parser(self, prog_name):
        parser = super(DeleteRegion, self).get_parser(prog_name)
        parser.add_argument('region', metavar='<region-id>', nargs='+', help=_('Region ID(s) to delete'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        result = 0
        for i in parsed_args.region:
            try:
                identity_client.regions.delete(i)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete region with ID '%(region)s': %(e)s"), {'region': i, 'e': e})
        if result > 0:
            total = len(parsed_args.region)
            msg = _('%(result)s of %(total)s regions failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)