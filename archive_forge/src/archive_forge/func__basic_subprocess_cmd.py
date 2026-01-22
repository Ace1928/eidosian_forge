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
def _basic_subprocess_cmd() -> Sequence[str]:
    import __main__
    if __main__.__spec__ is None:
        return [sys.executable, os.path.abspath(sys.argv[0])] + sys.argv[1:]
    return [sys.executable, '-m', __main__.__spec__.name] + sys.argv[1:]