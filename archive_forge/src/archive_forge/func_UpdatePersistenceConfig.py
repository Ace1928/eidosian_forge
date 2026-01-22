from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.command_lib.redis import cluster_util
from googlecloudsdk.command_lib.redis import util
def UpdatePersistenceConfig(unused_cluster_ref, args, patch_request):
    """Hook to add persistence config to the redis cluster update request."""
    if args.IsSpecified('persistence_mode') or args.IsSpecified('rdb_snapshot_period') or args.IsSpecified('rdb_snapshot_start_time') or args.IsSpecified('aof_append_fsync'):
        patch_request = AddFieldToUpdateMask('persistence_config', patch_request)
    if patch_request.cluster.persistenceConfig:
        if not args.IsSpecified('rdb_snapshot_period') and (not args.IsSpecified('rdb_snapshot_start_time')):
            patch_request.cluster.persistenceConfig.rdbConfig = None
        if not args.IsSpecified('aof_append_fsync'):
            patch_request.cluster.persistenceConfig.aofConfig = None
    return patch_request