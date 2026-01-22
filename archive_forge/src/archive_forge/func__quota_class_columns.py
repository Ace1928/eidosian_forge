from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
def _quota_class_columns(quota_class):
    return quota_class._info.keys()