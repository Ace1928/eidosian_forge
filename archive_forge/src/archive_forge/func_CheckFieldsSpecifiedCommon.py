from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.command_lib.redis import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.console import console_io
from six.moves import filter  # pylint: disable=redefined-builtin
def CheckFieldsSpecifiedCommon(args, patch_request, additional_update_args):
    """Checks fields to update that are registered for all tracks."""
    update_args = ['clear_labels', 'display_name', 'enable_auth', 'remove_labels', 'remove_redis_config', 'size', 'update_labels', 'update_redis_config', 'read_replicas_mode', 'secondary_ip_range', 'replica_count', 'persistence_mode', 'rdb_snapshot_period', 'rdb_snapshot_start_time', 'maintenance_window_day', 'maintenance_window_hour', 'maintenance_window_any'] + additional_update_args
    if list(filter(args.IsSpecified, update_args)):
        return patch_request
    raise NoFieldsSpecified('Must specify at least one valid instance parameter to update')