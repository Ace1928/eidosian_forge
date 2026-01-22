from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.fleet import client
from googlecloudsdk.command_lib.util.args import labels_util
def HandleNamespaceLabelsCreateRequest(ref, args, request):
    """Add namespace labels to create request.

  Args:
    ref: reference to the scope object.
    args: command line arguments.
    request: API request to be issued

  Returns:
    modified request

  """
    del ref
    release_track = args.calliope_command.ReleaseTrack()
    fleetclient = client.FleetClient(release_track)
    namespace_labels_diff = labels_util.Diff(additions=args.namespace_labels)
    ns_labels = namespace_labels_diff.Apply(fleetclient.messages.Scope.NamespaceLabelsValue, None).GetOrNone()
    request.scope.namespaceLabels = ns_labels
    return request