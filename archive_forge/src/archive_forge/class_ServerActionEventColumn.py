import logging
import uuid
from cliff import columns
import iso8601
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
class ServerActionEventColumn(columns.FormattableColumn):
    """Custom formatter for server action events.

    Format the :class:`~openstack.compute.v2.server_action.ServerActionEvent`
    objects as we'd like.
    """

    def _format_event(self, event):
        column_map = {}
        hidden_columns = ['id', 'name', 'location']
        _, columns = utils.get_osc_show_columns_for_sdk_resource(event, column_map, hidden_columns)
        data = utils.get_item_properties(event, columns)
        return dict(zip(columns, data))

    def human_readable(self):
        events = [self._format_event(event) for event in self._value]
        return utils.format_list_of_dicts(events)

    def machine_readable(self):
        events = [self._format_event(event) for event in self._value]
        return events