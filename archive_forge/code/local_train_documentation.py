from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import atexit
import json
import os
import subprocess
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from six.moves import range
Create a cluster configuration and start processes for the cluster.

  Args:
    module_name: str. Python module to use as the task.
    package_root: str. Absolute path to the package root of the module.
    num_ps: int. Number of parameter servers
    num_workers: int. Number of workers.
    num_evaluators: int. Number of evaluators.
    start_port: int. First port for the contiguous block of ports used
      by the cluster.
    user_args: [str]. Additional user args for the task. Any relative paths will
      not work.
  Returns:
    int. the retval of primary subprocess
  