from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from openstackclient.i18n import _
class ShowAccount(command.ShowOne):
    _description = _('Display account details')

    def take_action(self, parsed_args):
        data = self.app.client_manager.object_store.account_show()
        if 'properties' in data:
            data['properties'] = format_columns.DictColumn(data.pop('properties'))
        return zip(*sorted(data.items()))