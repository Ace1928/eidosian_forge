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
def _BuildJobsMappingDict(existing_jobs, project):
    """Builds a dictionary of unique jobs by attributes.

  Each key is in this dictionary is based on all the existing attributes of a
  job. Multiple jobs can map to the same key if all their attributes (schedule,
  url, timezone, description, etc.) match.

  Args:
    existing_jobs: A list of jobs that already exist in the backend. Each job
      maps to an apis.cloudscheduler.<ver>.cloudscheduler<ver>_messages.Job
      instance.
    project: The base name of the project.
  Returns:
    A dictionary where a key is built based on a all the job attributes and the
    value is an apis.cloudscheduler.<ver>.cloudscheduler<ver>_messages.Job
    instance.
  """
    jobs_indexed_dict = {}
    for job in existing_jobs:
        key = _CreateUniqueJobKeyForExistingJob(job, project)
        if key not in jobs_indexed_dict:
            jobs_indexed_dict[key] = []
        jobs_indexed_dict[key].append(job)
    return jobs_indexed_dict