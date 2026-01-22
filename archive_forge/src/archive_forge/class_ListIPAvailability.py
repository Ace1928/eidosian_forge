from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class ListIPAvailability(command.Lister):
    _description = _('List IP availability for network')

    def get_parser(self, prog_name):
        parser = super(ListIPAvailability, self).get_parser(prog_name)
        parser.add_argument('--ip-version', type=int, default=4, choices=[4, 6], metavar='<ip-version>', dest='ip_version', help=_('List IP availability of given IP version networks (default is 4)'))
        parser.add_argument('--project', metavar='<project>', help=_('List IP availability of given project (name or ID)'))
        identity_common.add_project_domain_option_to_parser(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        columns = ('network_id', 'network_name', 'total_ips', 'used_ips')
        column_headers = ('Network ID', 'Network Name', 'Total IPs', 'Used IPs')
        filters = {}
        if parsed_args.ip_version:
            filters['ip_version'] = parsed_args.ip_version
        if parsed_args.project:
            identity_client = self.app.client_manager.identity
            project_id = identity_common.find_project(identity_client, parsed_args.project, parsed_args.project_domain).id
            filters['project_id'] = project_id
        data = client.network_ip_availabilities(**filters)
        return (column_headers, (utils.get_item_properties(s, columns) for s in data))