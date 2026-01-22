from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.ai import operations
from googlecloudsdk.api_lib.ai.indexes import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai import flags
from googlecloudsdk.command_lib.ai import indexes_util
from googlecloudsdk.command_lib.ai import operations_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.GA)
class UpdateV1(base.UpdateCommand):
    """Update an Vertex AI index.

  ## EXAMPLES

  To update index `123` under project `example` in region `us-central1`, run:

    $ {command} --display-name=new-name
    --metadata-file=path/to/your/metadata.json --project=example
    --region=us-central1
  """

    @staticmethod
    def Args(parser):
        flags.AddIndexResourceArg(parser, 'to update')
        flags.GetDisplayNameArg('index', required=False).AddToParser(parser)
        flags.GetDescriptionArg('index').AddToParser(parser)
        flags.GetMetadataFilePathArg('index').AddToParser(parser)
        labels_util.AddUpdateLabelsFlags(parser)

    def _Run(self, args, version):
        index_ref = args.CONCEPTS.index.Parse()
        region = index_ref.AsDict()['locationsId']
        project_id = index_ref.AsDict()['projectsId']
        with endpoint_util.AiplatformEndpointOverrides(version, region=region):
            index_client = client.IndexesClient(version=version)
            if version == constants.GA_VERSION:
                operation = index_client.Patch(index_ref, args)
            else:
                operation = index_client.PatchBeta(index_ref, args)
            if args.metadata_file is not None:
                op_ref = indexes_util.ParseIndexOperation(operation.name)
                log.status.Print(constants.OPERATION_CREATION_DISPLAY_MESSAGE.format(name=operation.name, verb='update index', id=op_ref.Name(), sub_commands='--index={} --region={} [--project={}]'.format(index_ref.Name(), region, project_id)))
                return operation
            response_msg = operations_util.WaitForOpMaybe(operations_client=operations.OperationsClient(version=version), op=operation, op_ref=indexes_util.ParseIndexOperation(operation.name), log_method='update')
            if response_msg is not None:
                response = encoding.MessageToPyValue(response_msg)
                if 'name' in response:
                    log.UpdatedResource(response['name'], kind='Vertex AI index')
            return response_msg

    def Run(self, args):
        return self._Run(args, constants.GA_VERSION)