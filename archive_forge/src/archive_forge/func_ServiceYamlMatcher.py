from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.api_lib.app import yaml_parsing
from googlecloudsdk.command_lib.app import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
def ServiceYamlMatcher(path, stager, appyaml):
    """Generate a Service from an <service>.yaml source path.

  This function is a path matcher that returns if and only if:
  - `path` points to either a `<service>.yaml` or `<app-dir>` where
    `<app-dir>/app.yaml` exists.
  - the yaml-file is a valid <service>.yaml file.

  If the runtime and environment match an entry in the stager, the service will
  be staged into a directory.

  Args:
    path: str, Unsanitized absolute path, may point to a directory or a file of
        any type. There is no guarantee that it exists.
    stager: staging.Stager, stager that will be invoked if there is a runtime
        and environment match.
    appyaml: str or None, the app.yaml location to used for deployment.

  Raises:
    staging.StagingCommandFailedError, staging command failed.

  Returns:
    Service, fully populated with entries that respect a potentially
        staged deployable service, or None if the path does not match the
        pattern described.
  """
    descriptor = path if os.path.isfile(path) else os.path.join(path, 'app.yaml')
    _, ext = os.path.splitext(descriptor)
    if os.path.exists(descriptor) and ext in ['.yaml', '.yml']:
        app_dir = os.path.dirname(descriptor)
        service_info = yaml_parsing.ServiceYamlInfo.FromFile(descriptor)
        staging_dir = stager.Stage(descriptor, app_dir, service_info.runtime, service_info.env, appyaml)
        return Service(descriptor, app_dir, service_info, staging_dir or app_dir)
    return None