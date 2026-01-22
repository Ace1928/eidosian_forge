from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.datapipelines import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.datapipelines import flags
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class Describe(base.DescribeCommand):
    """Describe Data Pipelines Pipeline."""
    detailed_help = _DETAILED_HELP

    @staticmethod
    def Args(parser):
        flags.AddDescribePipelineFlags(parser)

    def Run(self, args):
        """Run the describe command."""
        client = util.PipelinesClient()
        pipelines_ref = args.CONCEPTS.pipeline.Parse()
        return client.Describe(pipeline=pipelines_ref.RelativeName())