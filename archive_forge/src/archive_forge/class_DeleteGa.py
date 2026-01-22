from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ai import operations
from googlecloudsdk.api_lib.ai.model_monitoring_jobs import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai import flags
from googlecloudsdk.command_lib.ai import model_monitoring_jobs_util
from googlecloudsdk.command_lib.ai import operations_util
from googlecloudsdk.core.console import console_io
@base.ReleaseTracks(base.ReleaseTrack.GA)
class DeleteGa(base.DeleteCommand):
    """Delete an existing Vertex AI model deployment monitoring job."""

    @staticmethod
    def Args(parser):
        flags.AddModelMonitoringJobResourceArg(parser, 'to delete')

    def Run(self, args):
        return _Run(args, constants.GA_VERSION)