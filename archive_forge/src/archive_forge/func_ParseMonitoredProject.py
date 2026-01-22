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
def ParseMonitoredProject(monitored_project_name, project_fallback):
    """Returns the metrics scope and monitored project.

  Parse the specified monitored project name and return the metrics scope and
  monitored project.

  Args:
    monitored_project_name: The name of the monitored project to create/delete.
    project_fallback: When set, allows monitored_project_name to be just a
      project id or number.

  Raises:
    MonitoredProjectNameError: If an invalid monitored project name is
    specified.

  Returns:
     (metrics_scope_def, monitored_project_def): Project parsed metrics scope
       project id, Project parsed metrics scope project id
  """
    matched = re.match('locations/global/metricsScopes/([a-z0-9:\\-]+)/projects/([a-z0-9:\\-]+)', monitored_project_name)
    if matched:
        if matched.group(0) != monitored_project_name:
            raise MonitoredProjectNameError('Invalid monitored project name has been specified.')
        metrics_scope_def = projects_util.ParseProject(matched.group(1))
        monitored_project_def = projects_util.ParseProject(matched.group(2))
    else:
        metrics_scope_def = projects_util.ParseProject(properties.VALUES.core.project.Get(required=True))
        monitored_resource_container_matched = re.match('projects/([a-z0-9:\\-]+)', monitored_project_name)
        if monitored_resource_container_matched:
            monitored_project_def = projects_util.ParseProject(monitored_resource_container_matched.group(1))
        elif project_fallback:
            log.warning('Received an incorrectly formatted project name. Expected "projects/{identifier}" received "{identifier}". Assuming given resource is a project.'.format(identifier=monitored_project_name))
            monitored_project_def = projects_util.ParseProject(monitored_project_name)
        else:
            raise MonitoredProjectNameError('Invalid monitored project name has been specified.')
    return (metrics_scope_def, monitored_project_def)