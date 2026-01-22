import os
import re
import subprocess
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import ClusterResolver
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import format_master_url
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.util.tf_export import tf_export
def _resolve_own_rank(self):
    """Returns the rank of the current task in range [0, num_tasks)."""
    return int(_get_slurm_var('PROCID'))