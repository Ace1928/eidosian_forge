from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataproc import dataproc as dp
from googlecloudsdk.api_lib.dataproc import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
def _GetJob(job_ref):
    return dataproc.client.projects_locations_batches.Get(dataproc.messages.DataprocProjectsLocationsBatchesGetRequest(name=job_ref))