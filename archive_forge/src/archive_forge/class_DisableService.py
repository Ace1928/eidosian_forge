from oslo_log import log as logging
from osc_lib.command import command
from osc_lib import utils
class DisableService(command.ShowOne):
    """Disable the Zun service."""
    log = logging.getLogger(__name__ + '.DisableService')

    def get_parser(self, prog_name):
        parser = super(DisableService, self).get_parser(prog_name)
        parser.add_argument('host', metavar='<host>', help='Name of host')
        parser.add_argument('binary', metavar='<binary>', help='Name of the binary to disable')
        parser.add_argument('--reason', metavar='<reason>', help='Reason for disabling service')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        host = parsed_args.host
        binary = parsed_args.binary
        reason = parsed_args.reason
        res = client.services.disable(host, binary, reason)
        columns = ('Host', 'Binary', 'Disabled', 'Disabled Reason')
        return (columns, utils.get_dict_properties(res[1]['service'], columns))