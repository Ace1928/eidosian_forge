from __future__ import annotations
import os
import subprocess
import sys
import warnings
from typing import Any, Optional, Sequence
def _spawn_daemon(args: Sequence[str]) -> None:
    """Spawn a daemon process (Unix)."""
    if sys.executable:
        _spawn_daemon_double_popen(args)
    else:
        _spawn(args)