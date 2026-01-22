from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ai.models import client
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai import flags
from googlecloudsdk.command_lib.ai import region_util
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class ListVersionV1Beta1(ListVersionV1):
    """List the model versions of the given region and model.

  ## EXAMPLES

  List the model version of a model `123` of project `example` in region
  `us-central1`, run:

    $ {command} 123 --project=example --region=us-central1
  """

    def _Run(self, args, model_ref, region):
        with endpoint_util.AiplatformEndpointOverrides(version=constants.BETA_VERSION, region=region):
            return client.ModelsClient().ListVersion(model_ref)