from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.fleet import client
from googlecloudsdk.api_lib.container.fleet import util as api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet import util
class ListBoundMemberships(base.ListCommand):
    """List memberships bound to a fleet scope.

  This command can fail for the following reasons:
  * The scope specified does not exist.
  * The user does not have access to the specified scope.

  ## EXAMPLES

  The following command lists memberships bound to scope `SCOPE` in
  project `PROJECT_ID`:

    $ {command} SCOPE --project=PROJECT_ID
  """

    @classmethod
    def Args(cls, parser):
        parser.display_info.AddFormat(util.MEM_LIST_FORMAT)
        resources.AddScopeResourceArg(parser, 'SCOPE', api_util.VERSION_MAP[cls.ReleaseTrack()], scope_help='Name of the fleet scope to list memberships bound to.', required=True)

    def Run(self, args):
        scope_arg = args.CONCEPTS.scope.Parse()
        scope_path = scope_arg.RelativeName()
        fleetclient = client.FleetClient(release_track=self.ReleaseTrack())
        return fleetclient.ListBoundMemberships(scope_path)