from oslo_log import log as logging
from osc_lib.command import command
from osc_lib import utils
class EnableService(command.ShowOne):
    """Enable the Zun service."""
    log = logging.getLogger(__name__ + '.EnableService')

    def get_parser(self, prog_name):
        parser = super(EnableService, self).get_parser(prog_name)
        parser.add_argument('host', metavar='<host>', help='Name of host')
        parser.add_argument('binary', metavar='<binary>', help='Name of the binary to enable')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        host = parsed_args.host
        binary = parsed_args.binary
        res = client.services.enable(host, binary)
        columns = ('Host', 'Binary', 'Disabled', 'Disabled Reason')
        return (columns, utils.get_dict_properties(res[1]['service'], columns))