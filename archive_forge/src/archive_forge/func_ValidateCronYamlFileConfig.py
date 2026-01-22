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
def ValidateCronYamlFileConfig(config):
    """Validates jobs configuration parameters in the cron YAML file.

  The purpose of this function is to mimick the behaviour of the old
  implementation of `gcloud app deploy cron.yaml` before migrating away
  from console-admin-hr. The errors generated are the same as the ones
  previously seen when gcloud sent the batch-request for updating jobs to the
  Zeus backend.

  Args:
     config: A yaml_parsing.ConfigYamlInfo object for the parsed YAML file we
      are going to process.

  Raises:
    HTTPError: Various different scenarios defined in the function can cause
      this exception to be raised.
  """
    cron_yaml = config.parsed
    if not cron_yaml.cron:
        return
    for job in cron_yaml.cron:
        if job.retry_parameters:
            if job.retry_parameters.job_retry_limit and job.retry_parameters.job_retry_limit > 5:
                _RaiseHTTPException('Invalid Cron retry parameters: Cannot set retry limit to more than 5 (currently set to {}).'.format(job.retry_parameters.job_retry_limit))
            if job.retry_parameters.job_age_limit and int(convertors.CheckAndConvertStringToFloatIfApplicable(job.retry_parameters.job_age_limit)) <= 0:
                _RaiseHTTPException('Invalid Cron retry parameters: Job age limit must be greater than zero seconds.')
            if job.retry_parameters.min_backoff_seconds is not None and job.retry_parameters.max_backoff_seconds is not None:
                min_backoff = job.retry_parameters.min_backoff_seconds
                max_backoff = job.retry_parameters.max_backoff_seconds
                if max_backoff < min_backoff:
                    _RaiseHTTPException('Invalid Cron retry parameters: Min backoff sec must not be greater than than max backoff sec.')