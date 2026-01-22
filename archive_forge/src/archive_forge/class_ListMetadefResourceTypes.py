from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
class ListMetadefResourceTypes(command.Lister):
    _description = _('List metadef resource types')

    def take_action(self, parsed_args):
        image_client = self.app.client_manager.image
        kwargs = {}
        data = image_client.metadef_resource_types(**kwargs)
        columns = ['Name']
        column_headers = columns
        return (column_headers, (utils.get_item_properties(s, columns) for s in data))