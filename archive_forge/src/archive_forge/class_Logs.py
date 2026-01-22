from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class Logs(base.Group):
    """Read logs for Cloud Run jobs executions."""
    detailed_help = {'EXAMPLES': '\n        To tail logs for a job execution, run:\n\n          $ {command} tail my-job-execution\n\n        To read logs for a job execution, run:\n\n          $ {command} read my-job-execution\n\n    '}