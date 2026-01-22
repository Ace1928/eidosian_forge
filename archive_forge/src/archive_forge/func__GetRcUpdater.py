from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import re
import shutil
import sys
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
import six
def _GetRcUpdater(completion_update, path_update, rc_path, sdk_root, host_os):
    """Returns an _RcUpdater object for the preferred user shell.

  Args:
    completion_update: bool, Whether or not to do command completion.
    path_update: bool, Whether or not to update PATH.
    rc_path: str, The path to the rc file to update. If None, ask.
    sdk_root: str, The path to the Cloud SDK root.
    host_os: str, The host os identification string.

  Returns:
    An _RcUpdater() object for the preferred user shell.
  """
    rc_path = _GetAndUpdateRcPath(completion_update, path_update, rc_path, host_os)
    preferred_shell = _GetPreferredShell(rc_path, default=_GetPreferredShell(encoding.GetEncodedValue(os.environ, 'SHELL', '/bin/sh')))
    return _RcUpdater(completion_update, path_update, preferred_shell, rc_path, sdk_root)