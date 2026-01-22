from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def BackupPoolArgument(required=True):
    return compute_flags.ResourceArgument(resource_name='backup pool', name='--backup-pool', completer=TargetPoolsCompleter, plural=False, required=required, regional_collection='compute.targetPools')