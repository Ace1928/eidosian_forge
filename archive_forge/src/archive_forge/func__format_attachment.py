import logging
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
def _format_attachment(attachment):
    columns = ('id', 'volume_id', 'instance', 'status', 'attach_mode', 'attached_at', 'detached_at', 'connection_info')
    column_headers = ('ID', 'Volume ID', 'Instance ID', 'Status', 'Attach Mode', 'Attached At', 'Detached At', 'Properties')
    if isinstance(attachment, dict):
        data = []
        for column in columns:
            if column == 'connection_info':
                data.append(format_columns.DictColumn(attachment[column]))
                continue
            data.append(attachment[column])
    else:
        data = utils.get_item_properties(attachment, columns, formatters={'connection_info': format_columns.DictColumn})
    return (column_headers, data)