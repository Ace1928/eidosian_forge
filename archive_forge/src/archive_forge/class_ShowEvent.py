import logging
import time
from cliff.formatters import base
from osc_lib.command import command
from osc_lib import utils
from heatclient._i18n import _
from heatclient.common import event_utils
from heatclient.common import utils as heat_utils
from heatclient import exc
class ShowEvent(command.ShowOne):
    """Show event details."""
    log = logging.getLogger(__name__ + '.ShowEvent')

    def get_parser(self, prog_name):
        parser = super(ShowEvent, self).get_parser(prog_name)
        parser.add_argument('stack', metavar='<stack>', help=_('Name or ID of stack to show events for'))
        parser.add_argument('resource', metavar='<resource>', help=_('Name of the resource event belongs to'))
        parser.add_argument('event', metavar='<event>', help=_('ID of event to display details for'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        client = self.app.client_manager.orchestration
        fields = {'stack_id': parsed_args.stack, 'resource_name': parsed_args.resource, 'event_id': parsed_args.event}
        try:
            client.stacks.get(parsed_args.stack)
            client.resources.get(parsed_args.stack, parsed_args.resource)
            event = client.events.get(**fields)
        except exc.HTTPNotFound as ex:
            raise exc.CommandError(str(ex))
        formatters = {'links': heat_utils.link_formatter, 'resource_properties': heat_utils.json_formatter}
        columns = []
        for key in event.to_dict():
            columns.append(key)
        return (columns, utils.get_item_properties(event, columns, formatters=formatters))