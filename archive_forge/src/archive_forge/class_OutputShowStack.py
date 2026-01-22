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
class OutputShowStack(command.ShowOne):
    """Show stack output."""
    log = logging.getLogger(__name__ + '.OutputShowStack')

    def get_parser(self, prog_name):
        parser = super(OutputShowStack, self).get_parser(prog_name)
        parser.add_argument('stack', metavar='<stack>', help=_('Name or ID of stack to query'))
        parser.add_argument('output', metavar='<output>', nargs='?', default=None, help=_('Name of an output to display'))
        parser.add_argument('--all', action='store_true', help=_('Display all stack outputs'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        client = self.app.client_manager.orchestration
        if not parsed_args.all and parsed_args.output is None:
            msg = _('Either <OUTPUT NAME> or --all must be specified.')
            raise exc.CommandError(msg)
        if parsed_args.all and parsed_args.output is not None:
            msg = _('Cannot specify both <OUTPUT NAME> and --all.')
            raise exc.CommandError(msg)
        if parsed_args.all:
            try:
                stack = client.stacks.get(parsed_args.stack)
            except heat_exc.HTTPNotFound:
                msg = _('Stack not found: %s') % parsed_args.stack
                raise exc.CommandError(msg)
            outputs = stack.to_dict().get('outputs', [])
            columns = []
            values = []
            for output in outputs:
                columns.append(output['output_key'])
                values.append(heat_utils.json_formatter(output))
            return (columns, values)
        try:
            output = client.stacks.output_show(parsed_args.stack, parsed_args.output)['output']
        except heat_exc.HTTPNotFound:
            msg = _('Stack %(id)s or output %(out)s not found.') % {'id': parsed_args.stack, 'out': parsed_args.output}
            try:
                output = None
                stack = client.stacks.get(parsed_args.stack).to_dict()
                for o in stack.get('outputs', []):
                    if o['output_key'] == parsed_args.output:
                        output = o
                        break
                if output is None:
                    raise exc.CommandError(msg)
            except heat_exc.HTTPNotFound:
                raise exc.CommandError(msg)
        if 'output_error' in output:
            msg = _('Output error: %s') % output['output_error']
            raise exc.CommandError(msg)
        return self.dict2columns(output)