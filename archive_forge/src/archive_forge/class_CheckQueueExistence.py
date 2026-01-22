import json
from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
from zaqarclient._i18n import _
from zaqarclient.transport import errors
class CheckQueueExistence(command.ShowOne):
    """Check queue existence"""
    _description = _('Check queue existence')
    log = logging.getLogger(__name__ + '.CheckQueueExistence')

    def get_parser(self, prog_name):
        parser = super(CheckQueueExistence, self).get_parser(prog_name)
        parser.add_argument('queue_name', metavar='<queue_name>', help='Name of the queue')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        queue_name = parsed_args.queue_name
        queue = client.queue(queue_name, auto_create=False)
        columns = ('Exists',)
        data = dict(exists=queue.exists())
        return (columns, utils.get_dict_properties(data, columns))