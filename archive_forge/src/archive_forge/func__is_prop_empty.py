import itertools
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
def _is_prop_empty(columns, props, prop_name):
    return True if not props[columns.index(prop_name)] else False