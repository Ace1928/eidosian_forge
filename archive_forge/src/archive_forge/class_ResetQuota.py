from cliff import lister
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from oslo_utils import uuidutils
from octaviaclient.osc.v2 import constants as const
from octaviaclient.osc.v2 import utils as v2_utils
class ResetQuota(command.Command):
    """Resets quotas to default quotas"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('project', metavar='<project>', help='Project to reset quotas (name or ID).')
        return parser

    def take_action(self, parsed_args):
        attrs = v2_utils.get_quota_attrs(self.app.client_manager, parsed_args)
        project_id = attrs.pop('project_id')
        self.app.client_manager.load_balancer.quota_reset(project_id=project_id)