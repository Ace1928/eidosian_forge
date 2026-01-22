from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import subprocess
import sys
from googlecloudsdk.command_lib.code import run_subprocess
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import times
import six
def _GetMinikubeDockerEnvs(cluster_name):
    """Get the docker environment settings for a given cluster."""
    cmd = [_FindMinikube(), 'docker-env', '-p', cluster_name, '--shell=none']
    lines = run_subprocess.GetOutputLines(cmd, timeout_sec=20)
    return dict((line.split('=', 1) for line in lines if line and (not line.startswith('#'))))