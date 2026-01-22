from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataproc import dataproc as dp
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dataproc import flags
@base.ReleaseTracks(base.ReleaseTrack.BETA)
@base.Hidden
class Analyze(base.Command):
    """Analyze a batch job."""
    detailed_help = {'EXAMPLES': '          To Analyze a batch job, run:\n\n            $ {command} my-batch-job --region=us-central1\n          '}

    @staticmethod
    def Args(parser):
        dataproc = dp.Dataproc(base.ReleaseTrack.GA)
        flags.AddBatchResourceArg(parser, 'analyze', dataproc.api_version, True)
        flags.AddTimeoutFlag(parser, default='30m')

    def Run(self, args):
        dataproc = dp.Dataproc(base.ReleaseTrack.GA)
        messages = dataproc.messages
        batch_id = args.CONCEPTS.batch.Parse()
        request = messages.DataprocProjectsLocationsBatchesAnalyzeRequest(name=batch_id.RelativeName())
        operation = dataproc.client.projects_locations_batches.Analyze(request)
        operation = util.WaitForOperation(dataproc, operation, message='Waiting for analyze batch', timeout_s=args.timeout)
        return operation