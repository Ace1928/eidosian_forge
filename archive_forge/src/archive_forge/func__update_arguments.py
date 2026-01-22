import copy
import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import tags as _tag
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
def _update_arguments(obj_list, parsed_args_list, option):
    for item in parsed_args_list:
        try:
            obj_list.remove(item)
        except ValueError:
            msg = _('Subnet does not contain %(option)s %(value)s') % {'option': option, 'value': item}
            raise exceptions.CommandError(msg)