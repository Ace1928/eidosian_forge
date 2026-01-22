from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
from zunclient.common import utils as zun_utils
from zunclient import exceptions as exc
from zunclient.i18n import _
class ShowRegistry(command.ShowOne):
    """Show a registry"""
    log = logging.getLogger(__name__ + '.ShowRegistry')

    def get_parser(self, prog_name):
        parser = super(ShowRegistry, self).get_parser(prog_name)
        parser.add_argument('registry', metavar='<registry>', help='ID or name of the registry to show.')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        opts = {}
        opts['id'] = parsed_args.registry
        opts = zun_utils.remove_null_parms(**opts)
        registry = client.registries.get(**opts)
        return zip(*sorted(registry._info['registry'].items()))