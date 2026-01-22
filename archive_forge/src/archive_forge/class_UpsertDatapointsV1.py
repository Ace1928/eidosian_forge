from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ai.indexes import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai import flags
@base.ReleaseTracks(base.ReleaseTrack.GA)
class UpsertDatapointsV1(base.CreateCommand):
    """Upsert data points into the specified index.

  ## EXAMPLES

  To upsert datapoints into an index '123', run:

    $ {command} 123 --datapoints-from-file=example.json
    --project=example --region=us-central1
  """

    @staticmethod
    def Args(parser):
        flags.AddIndexResourceArg(parser, 'to upsert data points from')
        flags.GetDatapointsFilePathArg('index', required=True).AddToParser(parser)
        flags.GetDynamicMetadataUpdateMaskArg(required=False).AddToParser(parser)

    def _Run(self, args, version):
        index_ref = args.CONCEPTS.index.Parse()
        region = index_ref.AsDict()['locationsId']
        with endpoint_util.AiplatformEndpointOverrides(version, region=region):
            index_client = client.IndexesClient(version=version)
            if version == constants.GA_VERSION:
                operation = index_client.UpsertDatapoints(index_ref, args)
            else:
                operation = index_client.UpsertDatapointsBeta(index_ref, args)
            return operation

    def Run(self, args):
        return self._Run(args, constants.GA_VERSION)