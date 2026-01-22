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
def _ValidateTaskRetryLimit(queue):
    """Validates task retry limit input values for both queues in the YAML file.

  Args:
    queue: third_party.appengine.api.queueinfo.QueueEntry, The QueueEntry
      instance generated from the parsed YAML file.

  Raises:
    HTTPError: Based on the inputs provided if value specified is negative.
  """
    if queue.retry_parameters.task_retry_limit and queue.retry_parameters.task_retry_limit < 0:
        _RaiseHTTPException('Invalid queue configuration. Task retry limit must not be less than zero.')