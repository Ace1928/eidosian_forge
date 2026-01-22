from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
import textwrap
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core.util import files
import six
def FindContainerPort(name):
    """Returns the port of the Docker container with specified name.

  Args:
    name: The name of the Docker container.

  Returns:
    str: The port number of the Docker container.

  Raises:
    DockerExecutionException: if the exit code of the execution is non-zero
    or if the port of the container does not exist.
  """
    mapping = '{{range $p, $conf := .NetworkSettings.Ports}}      {{(index $conf 0).HostPort}}{{end}}'
    find_port = [_DOCKER, 'inspect', '--format=' + mapping, name]
    out = []
    capture_out = lambda stdout: out.append(stdout.strip())
    status = execution_utils.Exec(find_port, out_func=capture_out, no_exit=True)
    if status:
        raise DockerExecutionException(status, 'Docker failed to execute: failed to find port for ' + name)
    return out[0]