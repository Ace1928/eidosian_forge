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
def RunKubectl(args):
    """Runs a kubectl command with the cluster referenced by this client.

  Args:
    args: command line arguments to pass to kubectl

  Returns:
    The contents of stdout if the return code is 0, stderr (or a fabricated
    error if stderr is empty) otherwise
  """
    cmd = [util.CheckKubectlInstalled()]
    cmd.extend(args)
    out = io.StringIO()
    err = io.StringIO()
    env = _GetEnvs()
    returncode = execution_utils.Exec(cmd, no_exit=True, out_func=out.write, err_func=err.write, in_str=None, env=env)
    if returncode != 0 and (not err.getvalue()):
        err.write('kubectl exited with return code {}'.format(returncode))
    return (out.getvalue() if returncode == 0 else None, err.getvalue() if returncode != 0 else None)