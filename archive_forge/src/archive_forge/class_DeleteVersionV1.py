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
@base.ReleaseTracks(base.ReleaseTrack.GA)
class DeleteVersionV1(base.DeleteCommand):
    """Delete an existing Vertex AI model version.

  ## EXAMPLES

  To delete a model `123` version `1234` under project `example` in region
  `us-central1`, run:

    $ {command} 123@1234 --project=example --region=us-central1
  """

    @staticmethod
    def Args(parser):
        flags.AddModelVersionResourceArg(parser, 'to delete', region_util.PromptForOpRegion)

    def _Run(self, args, model_version_ref, region):
        with endpoint_util.AiplatformEndpointOverrides(version=constants.GA_VERSION, region=region):
            client_instance = apis.GetClientInstance(constants.AI_PLATFORM_API_NAME, constants.AI_PLATFORM_API_VERSION[constants.GA_VERSION])
            return client.ModelsClient(client=client_instance, messages=client_instance.MESSAGES_MODULE).DeleteVersion(model_version_ref)

    def Run(self, args):
        model_version_ref = args.CONCEPTS.model_version.Parse()
        region = model_version_ref.AsDict()['locationsId']
        return self._Run(args, model_version_ref, region)