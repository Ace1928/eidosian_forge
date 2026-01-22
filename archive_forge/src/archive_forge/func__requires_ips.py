from __future__ import annotations
import os
import re
import socket
import subprocess
from subprocess import PIPE, Popen
from typing import Any, Callable, Iterable, Sequence
from warnings import warn
def _requires_ips(f: Callable) -> Callable:
    """decorator to ensure load_ips has been run before f"""

    def ips_loaded(*args: Any, **kwargs: Any) -> Any:
        _load_ips()
        return f(*args, **kwargs)
    return ips_loaded