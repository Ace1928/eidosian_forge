from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
from zunclient.common import template_utils
from zunclient.common import utils as zun_utils
from zunclient.i18n import _
def _capsule_columns(capsule):
    return capsule._info.keys()