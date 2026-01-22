import os
import re
import subprocess
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import ClusterResolver
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import format_master_url
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.util.tf_export import tf_export
def _get_slurm_var(name):
    """Gets the SLURM variable from the environment.

  Args:
    name: Name of the step variable

  Returns:
    SLURM_<name> from os.environ
  Raises:
    RuntimeError if variable is not found
  """
    name = 'SLURM_' + name
    try:
        return os.environ[name]
    except KeyError:
        raise RuntimeError('%s not found in environment. Not running inside a SLURM step?' % name)