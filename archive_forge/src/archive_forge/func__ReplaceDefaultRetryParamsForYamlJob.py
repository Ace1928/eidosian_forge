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
def _ReplaceDefaultRetryParamsForYamlJob(job):
    """Replaces default values for retry parameters.

  Retry parameters are set to their default values if not already user defined.
  These values are only set if the user has defined at least one retry
  parameter. Also we are limiting min_backoff to a minimum value of 5.0s since
  the new scheduler API does not support setting a lower value than this.
  Modifies input `job` argument directly.

  Args:
    job: An instance of a parsed YAML job object.
  """
    defaults = constants.CRON_JOB_LEGACY_DEFAULT_VALUES
    retry_data = job.retry_parameters
    if retry_data:
        if retry_data.min_backoff_seconds is None and retry_data.max_backoff_seconds is None:
            retry_data.min_backoff_seconds = defaults['min_backoff']
            retry_data.max_backoff_seconds = defaults['max_backoff']
        elif retry_data.min_backoff_seconds is None or retry_data.max_backoff_seconds is None:
            if not retry_data.min_backoff_seconds:
                retry_data.min_backoff_seconds = defaults['min_backoff']
            if retry_data.max_backoff_seconds:
                retry_data.min_backoff_seconds = min(retry_data.min_backoff_seconds, retry_data.max_backoff_seconds)
            if retry_data.max_backoff_seconds is None:
                retry_data.max_backoff_seconds = defaults['max_backoff']
            retry_data.max_backoff_seconds = max(retry_data.min_backoff_seconds, retry_data.max_backoff_seconds)
        if retry_data.max_doublings is None:
            retry_data.max_doublings = defaults['max_doublings']
        if retry_data.job_age_limit is None:
            retry_data.job_age_limit = defaults['max_retry_duration']