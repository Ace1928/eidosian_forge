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
def _launch_process_observer(child_processes: List[subprocess.Popen]) -> None:
    """Launches a thread that runs along the main process and monitors the health of all processes."""
    _ChildProcessObserver(child_processes=child_processes, main_pid=os.getpid()).start()