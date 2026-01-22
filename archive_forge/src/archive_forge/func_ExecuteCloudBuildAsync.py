from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import time
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.cloudbuild import logs as cloudbuild_logs
from googlecloudsdk.api_lib.util import requests
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from six.moves import range  # pylint: disable=redefined-builtin
def ExecuteCloudBuildAsync(self, build, project=None):
    """Execute a call to CloudBuild service and return the build operation.


    Args:
      build: Build object. The Build to execute.
      project: The project to execute, or None to use the current project
          property.

    Raises:
      BuildFailedError: when the build fails.

    Returns:
      build_op, an in-progress build operation.
    """
    if project is None:
        project = properties.VALUES.core.project.Get(required=True)
    build_op = self.client.projects_builds.Create(self.messages.CloudbuildProjectsBuildsCreateRequest(projectId=project, build=build))
    return build_op