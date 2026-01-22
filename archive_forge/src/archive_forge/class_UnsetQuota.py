from cliff import lister
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from oslo_utils import uuidutils
from octaviaclient.osc.v2 import constants as const
from octaviaclient.osc.v2 import utils as v2_utils
class UnsetQuota(command.Command):
    """Clear quota settings"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('project', metavar='<project>', help='Name or UUID of the project.')
        parser.add_argument('--loadbalancer', action='store_true', help='Reset the load balancer quota to the default.')
        parser.add_argument('--listener', action='store_true', help='Reset the listener quota to the default.')
        parser.add_argument('--pool', action='store_true', help='Reset the pool quota to the default.')
        parser.add_argument('--member', action='store_true', help='Reset the member quota to the default.')
        parser.add_argument('--healthmonitor', action='store_true', help='Reset the health monitor quota to the default.')
        parser.add_argument('--l7policy', action='store_true', help='Reset the l7policy quota to the default.')
        parser.add_argument('--l7rule', action='store_true', help='Reset the l7rule quota to the default.')
        return parser

    def take_action(self, parsed_args):
        unset_args = v2_utils.get_unsets(parsed_args)
        if not unset_args:
            return
        project_id = v2_utils.get_resource_id(self.app.client_manager.identity, 'project', parsed_args.project)
        body = {'quota': unset_args}
        self.app.client_manager.load_balancer.quota_set(project_id, json=body)