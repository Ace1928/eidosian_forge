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
def _check_can_spawn_children(self) -> None:
    if len(self.procs) > 0:
        raise RuntimeError('The launcher can only create subprocesses once.')
    if self.cluster_environment.local_rank() != 0:
        raise RuntimeError('Lightning attempted to launch new distributed processes with `local_rank > 0`. This should not happen. Possible reasons: 1) LOCAL_RANK environment variable was incorrectly modified by the user, 2) `ClusterEnvironment.creates_processes_externally` incorrectly implemented.')