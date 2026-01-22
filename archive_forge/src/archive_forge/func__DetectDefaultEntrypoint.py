from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
import subprocess
import textwrap
from gae_ext_runtime import ext_runtime
from googlecloudsdk.api_lib.app import ext_runtime_adapter
from googlecloudsdk.api_lib.app.images import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
def _DetectDefaultEntrypoint(path, gems):
    """Returns the app server expected by this application.

  Args:
    path: (str) Application path.
    gems: ([str, ...]) A list of gems used by this application.

  Returns:
    (str) The default entrypoint command, or the empty string if unknown.
  """
    procfile_path = os.path.join(path, 'Procfile')
    if os.path.isfile(procfile_path):
        return ENTRYPOINT_FOREMAN
    if 'puma' in gems:
        return ENTRYPOINT_PUMA
    elif 'unicorn' in gems:
        return ENTRYPOINT_UNICORN
    configru_path = os.path.join(path, 'config.ru')
    if os.path.isfile(configru_path):
        return ENTRYPOINT_RACKUP
    return ''