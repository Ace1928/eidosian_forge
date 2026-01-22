from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import io
import os
import re
import shutil
import tempfile
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.api_lib.app import runtime_registry
from googlecloudsdk.command_lib.app import jarfile
from googlecloudsdk.command_lib.util import java
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
class Stager(object):

    def __init__(self, registry, staging_area):
        self.registry = registry
        self.staging_area = staging_area

    def Stage(self, descriptor, app_dir, runtime, environment, appyaml=None):
        """Stage the given deployable or do nothing if N/A.

    Args:
      descriptor: str, path to the unstaged <service>.yaml or appengine-web.xml
      app_dir: str, path to the unstaged app directory
      runtime: str, the name of the runtime for the application to stage
      environment: api_lib.app.env.Environment, the environment for the
        application to stage
      appyaml: str or None, the app.yaml location to used for deployment.

    Returns:
      str, the path to the staged directory or None if no corresponding staging
          command was found.

    Raises:
      NoSdkRootError: if no Cloud SDK installation root could be found.
      StagingCommandFailedError: if the staging command process exited non-zero.
    """
        command = self.registry.Get(runtime, environment)
        if not command:
            return None
        command.EnsureInstalled()
        return command.Run(self.staging_area, descriptor, app_dir, appyaml)