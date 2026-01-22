import logging
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable, List, Optional, Sequence, Tuple
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override
from lightning_fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning_fabric.strategies.launchers.launcher import _Launcher
from lightning_fabric.utilities.distributed import _set_num_threads_if_needed
from lightning_fabric.utilities.rank_zero import rank_prefixed_message
def _hydra_subprocess_cmd(local_rank: int) -> Tuple[Sequence[str], str]:
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import get_original_cwd, to_absolute_path
    import __main__
    if __main__.__spec__ is None:
        command = [sys.executable, to_absolute_path(sys.argv[0])]
    else:
        command = [sys.executable, '-m', __main__.__spec__.name]
    command += sys.argv[1:]
    cwd = get_original_cwd()
    rundir = f'"{HydraConfig.get().run.dir}"'
    command += [f'hydra.run.dir={rundir}', f'hydra.job.name=train_ddp_process_{local_rank}', 'hydra.output_subdir=null']
    return (command, cwd)