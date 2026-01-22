from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
class ListAccessibleDomains(command.Lister):
    _description = _('List accessible domains')

    def take_action(self, parsed_args):
        columns = ('ID', 'Enabled', 'Name', 'Description')
        identity_client = self.app.client_manager.identity
        data = identity_client.federation.domains.list()
        return (columns, (utils.get_item_properties(s, columns, formatters={}) for s in data))