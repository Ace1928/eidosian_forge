import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.network import common
class DeleteFloatingIPPortForwarding(command.Command):
    _description = _('Delete floating IP port forwarding')

    def get_parser(self, prog_name):
        parser = super(DeleteFloatingIPPortForwarding, self).get_parser(prog_name)
        parser.add_argument('floating_ip', metavar='<floating-ip>', help=_('Floating IP that the port forwarding belongs to (IP address or ID)'))
        parser.add_argument('port_forwarding_id', nargs='+', metavar='<port-forwarding-id>', help=_('The ID of the floating IP port forwarding(s) to delete'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        floating_ip = client.find_ip(parsed_args.floating_ip, ignore_missing=False)
        result = 0
        for port_forwarding_id in parsed_args.port_forwarding_id:
            try:
                client.delete_floating_ip_port_forwarding(floating_ip.id, port_forwarding_id, ignore_missing=False)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete floating IP port forwarding '%(port_forwarding_id)s': %(e)s"), {'port_forwarding_id': port_forwarding_id, 'e': e})
        if result > 0:
            total = len(parsed_args.port_forwarding_id)
            msg = _('%(result)s of %(total)s Port forwarding failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)