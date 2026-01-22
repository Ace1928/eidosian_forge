from cliff import lister
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import tags as _tag
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
from octaviaclient.osc.v2 import constants as const
from octaviaclient.osc.v2 import utils as v2_utils
class SetLoadBalancer(command.Command):
    """Update a load balancer"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('loadbalancer', metavar='<load_balancer>', help='Name or UUID of the load balancer to update.')
        parser.add_argument('--name', metavar='<name>', help='Set load balancer name.')
        parser.add_argument('--description', metavar='<description>', help='Set load balancer description.')
        parser.add_argument('--vip-qos-policy-id', metavar='<vip_qos_policy_id>', help="Set QoS policy ID for VIP port. Unset with 'None'.")
        admin_group = parser.add_mutually_exclusive_group()
        admin_group.add_argument('--enable', action='store_true', default=None, help='Enable load balancer.')
        admin_group.add_argument('--disable', action='store_true', default=None, help='Disable load balancer.')
        parser.add_argument('--wait', action='store_true', help='Wait for action to complete.')
        _tag.add_tag_option_to_parser_for_set(parser, 'load balancer')
        return parser

    def take_action(self, parsed_args):
        attrs = v2_utils.get_loadbalancer_attrs(self.app.client_manager, parsed_args)
        lb_id = attrs.pop('loadbalancer_id')
        v2_utils.set_tags_for_set(self.app.client_manager.load_balancer.load_balancer_show, lb_id, attrs, clear_tags=parsed_args.no_tag)
        body = {'loadbalancer': attrs}
        self.app.client_manager.load_balancer.load_balancer_set(lb_id, json=body)
        if parsed_args.wait:
            v2_utils.wait_for_active(status_f=self.app.client_manager.load_balancer.load_balancer_show, res_id=lb_id)