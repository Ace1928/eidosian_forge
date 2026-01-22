from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from apitools.base.py import encoding
from googlecloudsdk.api_lib.ml_engine import jobs
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
def LogFilters(job_id, task_name=None):
    """Returns filters for log fetcher to use.

  Args:
    job_id: String id of job.
    task_name: String name of task.

  Returns:
    A list of filters to be passed to the logging API.
  """
    filters = ['(resource.type="ml_job" OR resource.type="cloudml_job")', 'resource.labels.job_id="{0}"'.format(job_id)]
    if task_name:
        filters.append('(resource.labels.task_name="{0}" OR labels.task_name="{0}")'.format(task_name))
    return filters