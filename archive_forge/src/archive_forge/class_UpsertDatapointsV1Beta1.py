from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ai.indexes import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai import flags
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class UpsertDatapointsV1Beta1(UpsertDatapointsV1):
    """Upsert data points into the specified index.

  ## EXAMPLES

  To upsert datapoints into an index `123`, run:

    $ {command} 123 --datapoints-from-file=example.json
    --project=example --region=us-central1
  """

    def Run(self, args):
        return self._Run(args, constants.BETA_VERSION)