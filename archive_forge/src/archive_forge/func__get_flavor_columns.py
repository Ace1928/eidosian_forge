import logging
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
def _get_flavor_columns(item):
    column_map = {'extra_specs': 'properties', 'ephemeral': 'OS-FLV-EXT-DATA:ephemeral', 'is_disabled': 'OS-FLV-DISABLED:disabled', 'is_public': 'os-flavor-access:is_public'}
    hidden_columns = ['links', 'location', 'original_name']
    return utils.get_osc_show_columns_for_sdk_resource(item, column_map, hidden_columns)