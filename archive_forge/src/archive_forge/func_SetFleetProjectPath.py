from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.edge_cloud.container import util
from googlecloudsdk.command_lib.run import flags
def SetFleetProjectPath(ref, args, request):
    """Sets the cluster.fleet.project field with a relative resource path.

  Args:
    ref: reference to the projectsId object.
    args: command line arguments.
    request: API request to be issued
  """
    release_track = args.calliope_command.ReleaseTrack()
    msgs = util.GetMessagesModule(release_track)
    if flags.FlagIsExplicitlySet(args, 'fleet_project'):
        request.cluster.fleet = msgs.Fleet()
        request.cluster.fleet.project = 'projects/' + args.fleet_project
    else:
        request.cluster.fleet = msgs.Fleet()
        request.cluster.fleet.project = 'projects/' + ref.projectsId