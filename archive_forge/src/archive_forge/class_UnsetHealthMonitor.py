from cliff import lister
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import tags as _tag
from oslo_utils import uuidutils
from octaviaclient.osc.v2 import constants as const
from octaviaclient.osc.v2 import utils as v2_utils
class UnsetHealthMonitor(command.Command):
    """Clear health monitor settings"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('health_monitor', metavar='<health_monitor>', help='Health monitor to update (name or ID).')
        parser.add_argument('--domain-name', action='store_true', help='Clear the health monitor domain name.')
        parser.add_argument('--expected-codes', action='store_true', help='Reset the health monitor expected codes to the API default.')
        parser.add_argument('--http-method', action='store_true', help='Reset the health monitor HTTP method to the API default.')
        parser.add_argument('--http-version', action='store_true', help='Reset the health monitor HTTP version to the API default.')
        parser.add_argument('--max-retries-down', action='store_true', help='Reset the health monitor max retries down to the API default.')
        parser.add_argument('--name', action='store_true', help='Clear the health monitor name.')
        parser.add_argument('--url-path', action='store_true', help='Clear the health monitor URL path.')
        parser.add_argument('--wait', action='store_true', help='Wait for action to complete.')
        _tag.add_tag_option_to_parser_for_unset(parser, 'health monitor')
        return parser

    def take_action(self, parsed_args):
        unset_args = v2_utils.get_unsets(parsed_args)
        if not unset_args and (not parsed_args.all_tag):
            return
        hm_id = v2_utils.get_resource_id(self.app.client_manager.load_balancer.health_monitor_list, 'healthmonitors', parsed_args.health_monitor)
        v2_utils.set_tags_for_unset(self.app.client_manager.load_balancer.health_monitor_show, hm_id, unset_args, clear_tags=parsed_args.all_tag)
        body = {'healthmonitor': unset_args}
        self.app.client_manager.load_balancer.health_monitor_set(hm_id, json=body)
        if parsed_args.wait:
            v2_utils.wait_for_active(status_f=self.app.client_manager.load_balancer.health_monitor_show, res_id=hm_id)