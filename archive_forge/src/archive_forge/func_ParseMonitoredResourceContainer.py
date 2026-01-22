from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.calliope import exceptions as calliope_exc
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
import six
def ParseMonitoredResourceContainer(monitored_resource_container_name, project_fallback):
    """Returns the monitored resource container identifier.

  Parse the specified monitored_resource_container_name and return the
  identifier.

  Args:
    monitored_resource_container_name: The monitored resource container. Ex -
      projects/12345.
    project_fallback: When set, allows monitored_resource_container_name to be
      just a project id or number.

  Raises:
    MonitoredProjectNameError: If an invalid monitored project name is
    specified.

  Returns:
     resource_type, monitored_resource_container_identifier: Monitored resource
     container type and identifier
  """
    matched = re.match('(projects)/([a-z0-9:\\-]+)', monitored_resource_container_name)
    if matched:
        return (matched.group(1), matched.group(2))
    elif project_fallback:
        log.warning('Received an incorrectly formatted project name. Expected "projects/{identifier}" received "{identifier}". Assuming given resource is a project.'.format(identifier=monitored_resource_container_name))
        return ('projects', projects_util.ParseProject(monitored_resource_container_name).Name())
    else:
        raise MonitoredProjectNameError('Invalid monitored project name has been specified.')