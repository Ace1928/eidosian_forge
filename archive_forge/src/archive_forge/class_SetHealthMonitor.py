from cliff import lister
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import tags as _tag
from oslo_utils import uuidutils
from octaviaclient.osc.v2 import constants as const
from octaviaclient.osc.v2 import utils as v2_utils
class SetHealthMonitor(command.Command):
    """Update a health monitor"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('health_monitor', metavar='<health_monitor>', help='Health monitor to update (name or ID).')
        parser.add_argument('--name', metavar='<name>', help='Set health monitor name.')
        parser.add_argument('--delay', metavar='<delay>', help='Set the time in seconds, between sending probes to members.')
        parser.add_argument('--domain-name', metavar='<domain_name>', help='Set the domain name, which be injected into the HTTP Host Header to the backend server for HTTP health check.')
        parser.add_argument('--expected-codes', metavar='<codes>', help='Set the list of HTTP status codes expected in response from the member to declare it healthy.')
        parser.add_argument('--http-method', metavar='{' + ','.join(HTTP_METHODS) + '}', choices=HTTP_METHODS, type=lambda s: s.upper(), help='Set the HTTP method that the health monitor uses for requests.')
        parser.add_argument('--http-version', metavar='<http_version>', choices=HTTP_VERSIONS, type=float, help='Set the HTTP version.')
        parser.add_argument('--timeout', metavar='<timeout>', help='Set the maximum time, in seconds, that a monitor waits to connect before it times out. This value must be less than the delay value.')
        parser.add_argument('--max-retries', metavar='<max_retries>', type=int, choices=range(1, 10), help='Set the number of successful checks before changing the operating status of the member to ONLINE.')
        parser.add_argument('--max-retries-down', metavar='<max_retries_down>', type=int, choices=range(1, 10), help='Set the number of allowed check failures before changing the operating status of the member to ERROR.')
        parser.add_argument('--url-path', metavar='<url_path>', help='Set the HTTP URL path of the request sent by the monitor to test the health of a backend member.')
        admin_group = parser.add_mutually_exclusive_group()
        admin_group.add_argument('--enable', action='store_true', default=None, help='Enable health monitor.')
        admin_group.add_argument('--disable', action='store_true', default=None, help='Disable health monitor.')
        parser.add_argument('--wait', action='store_true', help='Wait for action to complete.')
        _tag.add_tag_option_to_parser_for_set(parser, 'health monitor')
        return parser

    def take_action(self, parsed_args):
        attrs = v2_utils.get_health_monitor_attrs(self.app.client_manager, parsed_args)
        hm_id = attrs.pop('health_monitor_id')
        v2_utils.set_tags_for_set(self.app.client_manager.load_balancer.health_monitor_show, hm_id, attrs, clear_tags=parsed_args.no_tag)
        body = {'healthmonitor': attrs}
        self.app.client_manager.load_balancer.health_monitor_set(hm_id, json=body)
        if parsed_args.wait:
            v2_utils.wait_for_active(status_f=self.app.client_manager.load_balancer.health_monitor_show, res_id=hm_id)