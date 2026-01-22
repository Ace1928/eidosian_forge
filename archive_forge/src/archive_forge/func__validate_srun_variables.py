import logging
import os
import re
import shutil
import signal
import sys
from typing import Optional
from typing_extensions import override
from lightning_fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning_fabric.utilities.imports import _IS_WINDOWS
from lightning_fabric.utilities.rank_zero import rank_zero_warn
from lightning_fabric.utilities.warnings import PossibleUserWarning
@staticmethod
def _validate_srun_variables() -> None:
    """Checks for conflicting or incorrectly set variables set through `srun` and raises a useful error message.

        Right now, we only check for the most common user errors. See
        `the srun docs <https://slurm.schedmd.com/srun.html>`_
        for a complete list of supported srun variables.

        """
    ntasks = int(os.environ.get('SLURM_NTASKS', '1'))
    if ntasks > 1 and 'SLURM_NTASKS_PER_NODE' not in os.environ:
        raise RuntimeError(f'You set `--ntasks={ntasks}` in your SLURM bash script, but this variable is not supported. HINT: Use `--ntasks-per-node={ntasks}` instead.')