import logging
from osc_lib.command import command
from osc_lib import exceptions as exc
from osc_lib import utils
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient.common import format_utils
from heatclient.common import template_format
from heatclient.common import utils as heat_utils
from heatclient import exc as heat_exc
class CreateConfig(format_utils.JsonFormat):
    """Create software config"""
    log = logging.getLogger(__name__ + '.CreateConfig')

    def get_parser(self, prog_name):
        parser = super(CreateConfig, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<config-name>', help=_('Name of the software config to create'))
        parser.add_argument('--config-file', metavar='<config-file>', help=_('Path to JSON/YAML containing map defining <inputs>, <outputs>, and <options>'))
        parser.add_argument('--definition-file', metavar='<destination-file>', help=_('Path to software config script/data'))
        parser.add_argument('--group', metavar='<group>', default='Heat::Ungrouped', help=_('Group name of tool expected by the software config'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        heat_client = self.app.client_manager.orchestration
        return _create_config(heat_client, parsed_args)