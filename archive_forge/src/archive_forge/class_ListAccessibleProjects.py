from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
class ListAccessibleProjects(command.Lister):
    _description = _('List accessible projects')

    def take_action(self, parsed_args):
        columns = ('ID', 'Domain ID', 'Enabled', 'Name')
        identity_client = self.app.client_manager.identity
        data = identity_client.federation.projects.list()
        return (columns, (utils.get_item_properties(s, columns, formatters={}) for s in data))