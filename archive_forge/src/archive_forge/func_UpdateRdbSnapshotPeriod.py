from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.command_lib.redis import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.console import console_io
from six.moves import filter  # pylint: disable=redefined-builtin
def UpdateRdbSnapshotPeriod(unused_instance_ref, args, patch_request):
    """Hook to update RDB snapshot period."""
    if args.IsSpecified('rdb_snapshot_period'):
        patch_request = AddFieldToUpdateMask('persistence_config', patch_request)
    return patch_request