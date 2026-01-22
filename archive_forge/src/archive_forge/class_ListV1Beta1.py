from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ai.deployment_resource_pools import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai import flags
from googlecloudsdk.command_lib.ai import region_util
from googlecloudsdk.core import resources
@base.Hidden
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class ListV1Beta1(base.ListCommand):
    """List existing Vertex AI deployment resource pools.

  ## EXAMPLES

  To list the deployment resource pools under project ``example'' in region
  ``us-central1'', run:

    $ {command} --project=example --region=us-central1
  """

    @staticmethod
    def Args(parser):
        return _AddArgsBeta(parser)

    def Run(self, args):
        return _RunBeta(args)