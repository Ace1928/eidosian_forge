import logging
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ListQos(command.Lister):
    _description = _('List QoS specifications')

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        qos_specs_list = volume_client.qos_specs.list()
        for qos in qos_specs_list:
            try:
                qos_associations = volume_client.qos_specs.get_associations(qos)
                if qos_associations:
                    associations = [association.name for association in qos_associations]
                    qos._info.update({'associations': associations})
            except Exception as ex:
                if type(ex).__name__ == 'NotFound':
                    qos._info.update({'associations': None})
                else:
                    raise
        display_columns = ('ID', 'Name', 'Consumer', 'Associations', 'Properties')
        columns = ('ID', 'Name', 'Consumer', 'Associations', 'Specs')
        return (display_columns, (utils.get_dict_properties(s._info, columns, formatters={'Specs': format_columns.DictColumn, 'Associations': format_columns.ListColumn}) for s in qos_specs_list))