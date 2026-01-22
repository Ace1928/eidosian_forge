from __future__ import annotations
import os
import re
import socket
import subprocess
from subprocess import PIPE, Popen
from typing import Any, Callable, Iterable, Sequence
from warnings import warn
def _load_ips_gethostbyname() -> None:
    """load ip addresses with socket.gethostbyname_ex

    This can be slow.
    """
    global LOCALHOST
    try:
        LOCAL_IPS[:] = socket.gethostbyname_ex('localhost')[2]
    except OSError:
        LOCAL_IPS[:] = ['127.0.0.1']
    try:
        hostname = socket.gethostname()
        PUBLIC_IPS[:] = socket.gethostbyname_ex(hostname)[2]
        if not hostname.endswith('.local') and all((ip.startswith('127') for ip in PUBLIC_IPS)):
            PUBLIC_IPS[:] = socket.gethostbyname_ex(socket.gethostname() + '.local')[2]
    except OSError:
        pass
    finally:
        PUBLIC_IPS[:] = _uniq_stable(PUBLIC_IPS)
        LOCAL_IPS.extend(PUBLIC_IPS)
    LOCAL_IPS.extend(['0.0.0.0', ''])
    LOCAL_IPS[:] = _uniq_stable(LOCAL_IPS)
    LOCALHOST = LOCAL_IPS[0]