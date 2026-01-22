from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
def MakeSnapshotArg(plural=False):
    return compute_flags.ResourceArgument(resource_name='snapshot', name='snapshot_name', completer=compute_completers.RoutesCompleter, plural=plural, global_collection='compute.snapshots')