from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import fnmatch
import io
import json
import os
import re
import signal
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.command_lib.anthos.config.sync.common import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
def _GetEnvs():
    """Get the environment variables that should be passed to kubectl/gcloud commands.

  Returns:
    The dictionary that includes the environment varialbes.
  """
    env = dict(os.environ)
    if _KUBECONFIGENV not in env:
        env[_KUBECONFIGENV] = files.ExpandHomeDir(os.path.join('~', '.kube', _DEFAULTKUBECONFIG))
    return env