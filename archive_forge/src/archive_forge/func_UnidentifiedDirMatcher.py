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
def UnidentifiedDirMatcher(path, stager, appyaml):
    """Points out to the user that they need an app.yaml to deploy.

  Args:
    path: str, Unsanitized absolute path, may point to a directory or a file of
        any type. There is no guarantee that it exists.
    stager: staging.Stager, stager that will not be invoked.
    appyaml: str or None, the app.yaml location to used for deployment.
  Returns:
    None
  """
    del stager, appyaml
    if os.path.isdir(path):
        log.error(NO_YAML_ERROR)
    return None