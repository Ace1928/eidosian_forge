from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.datapipelines import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.datapipelines import flags
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class Create(base.CreateCommand):
    """Creates Data Pipelines Pipeline."""
    detailed_help = _DETAILED_HELP

    @staticmethod
    def Args(parser):
        flags.AddCreatePipelineFlags(parser)
        flags.GetDisplayNameArg('Data Pipelines pipeline').AddToParser(parser)
        flags.GetPipelineTypeArg(required=True).AddToParser(parser)
        flags.GetTemplateTypeArg(required=False).AddToParser(parser)
        flags.GetScheduleArg(required=False).AddToParser(parser)
        flags.GetTimeZoneArg(required=False).AddToParser(parser)
        flags.GetTemplateFileGcsLocationArg(required=False).AddToParser(parser)
        flags.GetParametersArg(required=False).AddToParser(parser)
        flags.GetMaxWorkersArg(required=False).AddToParser(parser)
        flags.GetNumWorkersArg(required=False).AddToParser(parser)
        flags.GetNetworkArg(required=False).AddToParser(parser)
        flags.GetSubnetworkArg(required=False).AddToParser(parser)
        flags.GetWorkerMachineTypeArg(required=False).AddToParser(parser)
        flags.GetTempLocationArg(required=False).AddToParser(parser)
        flags.GetDataflowKmsKeyArg(required=False).AddToParser(parser)
        flags.GetDisablePublicIpsArg(required=False).AddToParser(parser)
        flags.GetDataflowServiceAccountEmailArg(required=False).AddToParser(parser)
        flags.GetEnableStreamingEngineArg(required=False).AddToParser(parser)
        flags.GetAdditionalExperimentsArg(required=False).AddToParser(parser)
        flags.GetAdditionalUserLabelsArg(required=False).AddToParser(parser)
        flags.GetWorkerRegionArgs(required=False).AddToParser(parser)
        flags.GetFlexRsGoalArg(required=False).AddToParser(parser)
        flags.GetStreamingUpdateArgs(required=False).AddToParser(parser)

    def Run(self, args):
        """Run the create command."""
        client = util.PipelinesClient()
        pipelines_ref = args.CONCEPTS.pipeline.Parse()
        region_ref = pipelines_ref.Parent()
        return client.Create(pipeline=pipelines_ref.RelativeName(), parent=region_ref.RelativeName(), args=args)