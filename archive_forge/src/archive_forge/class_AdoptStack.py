import logging
import sys
from osc_lib.command import command
from osc_lib import exceptions as exc
from osc_lib import utils
from oslo_serialization import jsonutils
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient.common import event_utils
from heatclient.common import format_utils
from heatclient.common import hook_utils
from heatclient.common import http
from heatclient.common import template_utils
from heatclient.common import utils as heat_utils
from heatclient import exc as heat_exc
class AdoptStack(command.ShowOne):
    """Adopt a stack."""
    log = logging.getLogger(__name__ + '.AdoptStack')

    def get_parser(self, prog_name):
        parser = super(AdoptStack, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<stack-name>', help=_('Name of the stack to adopt'))
        parser.add_argument('-e', '--environment', metavar='<environment>', action='append', help=_('Path to the environment. Can be specified multiple times'))
        parser.add_argument('--timeout', metavar='<timeout>', type=int, help=_('Stack creation timeout in minutes'))
        parser.add_argument('--enable-rollback', action='store_true', help=_('Enable rollback on create/update failure'))
        parser.add_argument('--parameter', metavar='<key=value>', action='append', help=_('Parameter values used to create the stack. Can be specified multiple times'))
        parser.add_argument('--wait', action='store_true', help=_('Wait until stack adopt completes'))
        parser.add_argument('--adopt-file', metavar='<adopt-file>', required=True, help=_('Path to adopt stack data file'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        client = self.app.client_manager.orchestration
        env_files, env = template_utils.process_multiple_environments_and_files(env_paths=parsed_args.environment)
        adopt_url = heat_utils.normalise_file_path_to_url(parsed_args.adopt_file)
        adopt_data = request.urlopen(adopt_url).read().decode('utf-8')
        yaml_adopt_data = yaml.safe_load(adopt_data) or {}
        files = yaml_adopt_data.get('files', {})
        files.update(env_files)
        fields = {'stack_name': parsed_args.name, 'disable_rollback': not parsed_args.enable_rollback, 'adopt_stack_data': adopt_data, 'parameters': heat_utils.format_parameters(parsed_args.parameter), 'files': files, 'environment': env, 'timeout': parsed_args.timeout}
        stack = client.stacks.create(**fields)['stack']
        if parsed_args.wait:
            stack_status, msg = event_utils.poll_for_events(client, parsed_args.name, action='ADOPT')
            if stack_status == 'ADOPT_FAILED':
                raise exc.CommandError(msg)
        return _show_stack(client, stack['id'], format='table', short=True)