from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
class ListNetworkServiceProvider(command.Lister):
    _description = _('List Service Providers')

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        columns = ('service_type', 'name', 'is_default')
        column_headers = ('Service Type', 'Name', 'Default')
        data = client.service_providers()
        return (column_headers, (utils.get_item_properties(s, columns) for s in data))