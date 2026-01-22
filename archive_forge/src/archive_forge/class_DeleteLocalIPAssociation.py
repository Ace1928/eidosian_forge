import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class DeleteLocalIPAssociation(command.Command):
    _description = _('Delete Local IP association(s)')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('local_ip', metavar='<local-ip>', help=_('Local IP that the port association belongs to (Name or ID)'))
        parser.add_argument('fixed_port_id', nargs='+', metavar='<fixed-port-id>', help=_('The fixed port ID of Local IP Association'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        local_ip = client.find_local_ip(parsed_args.local_ip, ignore_missing=False)
        result = 0
        for fixed_port_id in parsed_args.fixed_port_id:
            try:
                client.delete_local_ip_association(local_ip.id, fixed_port_id, ignore_missing=False)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete Local IP Association with fixed port name or ID '%(fixed_port_id)s': %(e)s"), {'fixed_port_id': fixed_port_id, 'e': e})
        if result > 0:
            total = len(parsed_args.fixed_port_id)
            msg = _('%(result)s of %(total)s Local IP Associations failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)