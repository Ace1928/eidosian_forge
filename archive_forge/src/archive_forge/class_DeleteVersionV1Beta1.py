from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ai import operations
from googlecloudsdk.api_lib.ai.models import client
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai import flags
from googlecloudsdk.command_lib.ai import models_util
from googlecloudsdk.command_lib.ai import operations_util
from googlecloudsdk.command_lib.ai import region_util
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class DeleteVersionV1Beta1(DeleteVersionV1):
    """Delete an existing Vertex AI model version.

  ## EXAMPLES

  To delete a model `123` of version `1234` under project `example` in region
  `us-central1`, run:

    $ {command} 123@1234 --project=example --region=us-central1
  """

    def _Run(self, args, model_version_ref, region):
        with endpoint_util.AiplatformEndpointOverrides(version=constants.BETA_VERSION, region=region):
            operation = client.ModelsClient().DeleteVersion(model_version_ref)
            return operations_util.WaitForOpMaybe(operations_client=operations.OperationsClient(), op=operation, op_ref=models_util.ParseModelOperation(operation.name))