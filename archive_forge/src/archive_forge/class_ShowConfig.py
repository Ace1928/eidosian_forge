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
class ShowConfig(format_utils.YamlFormat):
    """Show software config details"""
    log = logging.getLogger(__name__ + '.ShowConfig')

    def get_parser(self, prog_name):
        parser = super(ShowConfig, self).get_parser(prog_name)
        parser.add_argument('config', metavar='<config>', help=_('ID of the config'))
        parser.add_argument('--config-only', default=False, action='store_true', help=_('Only display the value of the <config> property.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        heat_client = self.app.client_manager.orchestration
        return _show_config(heat_client, config_id=parsed_args.config, config_only=parsed_args.config_only)