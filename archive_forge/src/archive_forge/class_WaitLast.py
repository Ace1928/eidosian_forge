from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.workflows import workflows
from googlecloudsdk.calliope import base
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class WaitLast(base.DescribeCommand):
    """Wait for the last cached workflow execution to complete."""
    detailed_help = {'DESCRIPTION': '{description}', 'EXAMPLES': '        To wait for the last cached workflow execution to complete, run:\n\n          $ {command}\n        '}

    def Run(self, args):
        """Starts the wait on the completion of the execution."""
        api_version = workflows.ReleaseTrackToApiVersion(self.ReleaseTrack())
        client = workflows.WorkflowExecutionClient(api_version)
        return client.WaitForExecution(None)