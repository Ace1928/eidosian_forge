import os
import re
import subprocess
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import ClusterResolver
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import format_master_url
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.util.tf_export import tf_export
def _get_num_slurm_tasks():
    """Returns the number of SLURM tasks of the current job step.

  Returns:
    The number of tasks as an int
  """
    return int(_get_slurm_var('STEP_NUM_TASKS'))