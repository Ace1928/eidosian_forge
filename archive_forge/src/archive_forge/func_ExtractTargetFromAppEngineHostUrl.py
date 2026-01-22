from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.command_lib.tasks import app
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import http_encoding
import six  # pylint: disable=unused-import
from six.moves import filter  # pylint:disable=redefined-builtin
from six.moves import map  # pylint:disable=redefined-builtin
def ExtractTargetFromAppEngineHostUrl(job, project):
    """Extracts any target (service) if it exists in the appEngineRouting field.

  Args:
    job: An instance of job fetched from the backend.
    project: The base name of the project.

  Returns:
    The target if it exists in the URL, or if it is present in the service
    attribute of the appEngineRouting field, returns None otherwise.
    Some examples are:
      'alpha.some_project.uk.r.appspot.com' => 'alpha'
      'some_project.uk.r.appspot.com' => None
  """
    target = None
    try:
        target = job.appEngineHttpTarget.appEngineRouting.service
    except AttributeError:
        pass
    if target:
        return target
    host_url = None
    try:
        host_url = job.appEngineHttpTarget.appEngineRouting.host
    except AttributeError:
        pass
    if not host_url:
        return None
    delimiter = '.{}.'.format(project)
    return host_url.split(delimiter, 1)[0] if delimiter in host_url else None