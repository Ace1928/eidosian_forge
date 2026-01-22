from __future__ import annotations
import os
import re
import socket
import subprocess
from subprocess import PIPE, Popen
from typing import Any, Callable, Iterable, Sequence
from warnings import warn
def _load_ips_dumb() -> None:
    """Fallback in case of unexpected failure"""
    global LOCALHOST
    LOCALHOST = '127.0.0.1'
    LOCAL_IPS[:] = [LOCALHOST, '0.0.0.0', '']
    PUBLIC_IPS[:] = []