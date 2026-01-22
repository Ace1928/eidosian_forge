from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.edge_cloud.container import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.run import flags
def RemoveMaintenanceExclusionWindow(ref, args, request):
    """Removes the cluster.maintenance_policy.maintenance_exclusion_window if --remove-maintenance-exclusion-window flag is specified.

  Args:
    ref: reference to the cluster object.
    args: command line arguments.
    request: API request to be issued

  Returns:
    modified request
  """
    del ref
    if not flags.FlagIsExplicitlySet(args, 'remove_maintenance_exclusion_window'):
        return request
    if request.cluster is None:
        release_track = args.calliope_command.ReleaseTrack()
        request.cluster = util.GetMessagesModule(release_track).Cluster()
    if request.cluster.maintenancePolicy is None:
        _AddFieldToUpdateMask('maintenancePolicy', request)
        return request
    for idx, mew in enumerate(request.cluster.maintenancePolicy.maintenanceExclusions):
        if mew.id == args.remove_maintenance_exclusion_window:
            i = idx
            break
    else:
        raise exceptions.BadArgumentException('--remove-maintenance-exclusion-window', "Cannot remove a maintenance exclusion window that doesn't exist.")
    del request.cluster.maintenancePolicy.maintenanceExclusions[i]
    _AddFieldToUpdateMask('maintenancePolicy', request)
    return request