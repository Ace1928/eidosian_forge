from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import os
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import exceptions as c_except
def ParseCleanupArgs(self, job_id, data_shard_ids=None, target_profile=None, datastream=False, dataflow=False, pub_sub=False, monitoring=False, log_level=None, **kwargs):
    """"Parse args for the cleanup command."""
    del kwargs
    exec_args = ['cleanup']
    if job_id:
        exec_args.extend(['--jobId', job_id])
    if data_shard_ids:
        exec_args.extend(['--dataShardIds', data_shard_ids])
    if target_profile:
        exec_args.extend(['--target-profile', target_profile])
    if datastream:
        exec_args.append('--datastream')
    if dataflow:
        exec_args.append('--dataflow')
    if pub_sub:
        exec_args.append('--pubsub')
    if monitoring:
        exec_args.append('--monitoring')
    if log_level:
        exec_args.append('--log-level')
    return exec_args