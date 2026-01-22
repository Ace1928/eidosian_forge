from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.ai.custom_jobs import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.ai import constants
from googlecloudsdk.command_lib.ai import endpoint_util
from googlecloudsdk.command_lib.ai import validation as common_validation
from googlecloudsdk.command_lib.ai.custom_jobs import custom_jobs_util
from googlecloudsdk.command_lib.ai.custom_jobs import flags
from googlecloudsdk.command_lib.ai.custom_jobs import validation
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def _DisplayResult(self, response):
    cmd_prefix = 'gcloud'
    if self.ReleaseTrack().prefix:
        cmd_prefix += ' ' + self.ReleaseTrack().prefix
    log.status.Print(_JOB_CREATION_DISPLAY_MESSAGE_TEMPLATE.format(job_name=response.name, command_prefix=cmd_prefix))