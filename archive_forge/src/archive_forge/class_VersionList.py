import logging
from osc_lib.command import command
from osc_lib import utils
from heatclient._i18n import _
from heatclient.common import format_utils
from heatclient.common import http
from heatclient.common import template_utils
from heatclient.common import utils as heat_utils
from heatclient import exc
class VersionList(command.Lister):
    """List the available template versions."""
    log = logging.getLogger(__name__ + '.VersionList')

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        client = self.app.client_manager.orchestration
        versions = client.template_versions.list()
        try:
            versions[1].aliases

            def format_alias(aliases):
                return ','.join(aliases)
            fields = ['Version', 'Type', 'Aliases']
            formatters = {'Aliases': format_alias}
        except AttributeError:
            fields = ['Version', 'Type']
            formatters = None
        items = (utils.get_item_properties(s, fields, formatters=formatters) for s in versions)
        return (fields, items)