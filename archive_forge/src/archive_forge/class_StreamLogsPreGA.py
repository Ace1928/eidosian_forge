from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ai.custom_jobs import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai import flags as common_flags
from googlecloudsdk.command_lib.ai import log_util
from googlecloudsdk.command_lib.ai.custom_jobs import flags as custom_job_flags
from googlecloudsdk.command_lib.ai.custom_jobs import validation
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class StreamLogsPreGA(StreamLogsGA):
    """Show stream logs from a running custom job.

    ## EXAMPLES

    To stream logs of custom job ``123'' under project ``example'' in region
    ``us-central1'', run:

      $ {command} 123 --project=example --region=us-central1
  """
    _api_version = constants.BETA_VERSION