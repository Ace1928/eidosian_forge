from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.eventarc import common
from googlecloudsdk.api_lib.services import serviceusage
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import retry
def _GetOrCreateP4SA(service_name):
    """Gets (or creates) the P4SA for Eventarc in the given project.

  If the P4SA does not exist for this project, it will be created. Otherwise,
  the email address of the existing P4SA will be returned.

  Args:
    service_name: str, name of the service for the P4SA, e.g.
      eventarc.googleapis.com

  Returns:
    Email address of the Eventarc P4SA for the given project.
  """
    project_name = properties.VALUES.core.project.Get(required=True)
    response = serviceusage.GenerateServiceIdentity(project_name, service_name)
    return response['email']