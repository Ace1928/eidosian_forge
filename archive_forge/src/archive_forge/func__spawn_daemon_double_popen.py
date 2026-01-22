from __future__ import annotations
import os
import subprocess
import sys
import warnings
from typing import Any, Optional, Sequence
def _spawn_daemon_double_popen(args: Sequence[str]) -> None:
    """Spawn a daemon process using a double subprocess.Popen."""
    spawner_args = [sys.executable, _THIS_FILE]
    spawner_args.extend(args)
    temp_proc = subprocess.Popen(spawner_args, close_fds=True)
    _popen_wait(temp_proc, _WAIT_TIMEOUT)