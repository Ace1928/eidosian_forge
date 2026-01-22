from oslo_log import log as logging
from osc_lib.command import command
from osc_lib import utils
class DeleteService(command.Command):
    """Delete the Zun binaries/services."""
    log = logging.getLogger(__name__ + '.DeleteService')

    def get_parser(self, prog_name):
        parser = super(DeleteService, self).get_parser(prog_name)
        parser.add_argument('host', metavar='<host>', help='Name of host')
        parser.add_argument('binary', metavar='<binary>', help='Name of the binary to delete')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        host = parsed_args.host
        binary = parsed_args.binary
        try:
            client.services.delete(host, binary)
            print('Request to delete binary %s on host %s has been accepted.' % (binary, host))
        except Exception as e:
            print('Delete for binary %s on host %s failed: %s' % (binary, host, e))