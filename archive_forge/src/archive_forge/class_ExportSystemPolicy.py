from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.binauthz import apis
from googlecloudsdk.api_lib.container.binauthz import system_policy
from googlecloudsdk.api_lib.container.binauthz import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container.binauthz import arg_parsers
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class ExportSystemPolicy(base.Command):
    """Export the Binary Authorization system policy.

  For reliability reasons, the system policy is updated one region at a time.
  Because of this precaution, the system policy can differ between regions
  during an update. Use --location to view the system policy of a specific
  region.

  If --location is not specified, an arbitrary region is used. (Specifically, a
  region in the last group of regions to receive updates. Since most changes are
  additions, this will show the minimal set of system images that are allowed
  in all regions.)

  ## EXAMPLES

  To view the system policy:

      $ {command}

  To view the system policy in the region us-central1:

      $ {command} --location=us-central1
  """

    @classmethod
    def Args(cls, parser):
        parser.add_argument('--location', choices=arg_parsers.BINAUTHZ_ENFORCER_REGIONS, required=False, default='global', help='The region for which to get the system policy (or "global").')

    def Run(self, args):
        api_version = apis.GetApiVersion(self.ReleaseTrack())
        return system_policy.Client(api_version).Get(util.GetSystemPolicyRef(args.location))