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
def CreateJobInstance(scheduler_api, yaml_job):
    """Build a proto format job instance  matching the input YAML based job.

  Args:
    scheduler_api: api_lib.scheduler.<Alpha|Beta|GA>ApiAdapter, Cloud Scheduler
      API needed for doing jobs based operations.
    yaml_job: A parsed yaml_job entry read from the 'cron.yaml' file.
  Returns:
    An cloudscheduler.<ver>.cloudscheduler_<ver>_messages.Job instance.
  """
    messages = scheduler_api.messages
    if yaml_job.retry_parameters:
        retry_config = messages.RetryConfig(maxBackoffDuration=convertors.ConvertBackoffSeconds(yaml_job.retry_parameters.max_backoff_seconds), maxDoublings=yaml_job.retry_parameters.max_doublings, maxRetryDuration=convertors.ConvertTaskAgeLimit(yaml_job.retry_parameters.job_age_limit), minBackoffDuration=convertors.ConvertBackoffSeconds(yaml_job.retry_parameters.min_backoff_seconds), retryCount=yaml_job.retry_parameters.job_retry_limit)
    else:
        retry_config = None
    return messages.Job(appEngineHttpTarget=messages.AppEngineHttpTarget(httpMethod=messages.AppEngineHttpTarget.HttpMethodValueValuesEnum.GET, relativeUri=yaml_job.url, appEngineRouting=messages.AppEngineRouting(service=yaml_job.target)), retryConfig=retry_config, description=yaml_job.description, legacyAppEngineCron=scheduler_api.jobs.legacy_cron, schedule=yaml_job.schedule, timeZone=yaml_job.timezone if yaml_job.timezone else 'UTC')