import logging
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
def _format_group_type(group):
    columns = ('id', 'name', 'description', 'is_public', 'group_specs')
    column_headers = ('ID', 'Name', 'Description', 'Is Public', 'Properties')
    return (column_headers, utils.get_item_properties(group, columns, formatters={'group_specs': format_columns.DictColumn}))