from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.app import util
from googlecloudsdk.api_lib.tasks import task_queues_convertors as convertors
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.tasks import app
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.command_lib.tasks import flags
from googlecloudsdk.command_lib.tasks import parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
import six
from six.moves import urllib
def _CreateUniqueJobKeyForExistingJob(job, project):
    """Creates a key from the proto job instance's attributes passed as input.

  Args:
    job: An instance of job fetched from the backend.
    project: The base name of the project.
  Returns:
    A tuple of attributes used as a key to identify this job.
  """
    return (job.schedule, job.timeZone, job.appEngineHttpTarget.relativeUri, job.description, convertors.CheckAndConvertStringToFloatIfApplicable(job.retryConfig.minBackoffDuration) if job.retryConfig else None, convertors.CheckAndConvertStringToFloatIfApplicable(job.retryConfig.maxBackoffDuration) if job.retryConfig else None, job.retryConfig.maxDoublings if job.retryConfig else None, convertors.CheckAndConvertStringToFloatIfApplicable(job.retryConfig.maxRetryDuration) if job.retryConfig else None, job.retryConfig.retryCount if job.retryConfig else None, parsers.ExtractTargetFromAppEngineHostUrl(job, project))