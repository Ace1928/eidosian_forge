import datetime
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from typing import TYPE_CHECKING, Any, Dict, Optional
from wandb import _sentry, termlog
from wandb.env import error_reporting_enabled
from wandb.errors import Error
from wandb.sdk.lib.wburls import wburls
from wandb.util import get_core_path, get_module
from . import _startup_debug, port_file
from .service_base import ServiceInterface
from .service_sock import ServiceSockInterface
def _wait_for_ports(self, fname: str, proc: Optional[subprocess.Popen]=None) -> None:
    """Wait for the service to write the port file and then read it.

        Args:
            fname: The path to the port file.
            proc: The process to wait for.

        Raises:
            ServiceStartTimeoutError: If the service takes too long to start.
            ServiceStartPortError: If the service writes an invalid port file or unable to read it.
            ServiceStartProcessError: If the service process exits unexpectedly.

        """
    time_max = time.monotonic() + self._settings._service_wait
    while time.monotonic() < time_max:
        if proc and proc.poll():
            context = dict(command=proc.args, sys_executable=sys.executable, which_python=shutil.which('python3'), proc_out=proc.stdout.read() if proc.stdout else '', proc_err=proc.stderr.read() if proc.stderr else '')
            raise ServiceStartProcessError(f'The wandb service process exited with {proc.returncode}. Ensure that `sys.executable` is a valid python interpreter. You can override it with the `_executable` setting or with the `WANDB__EXECUTABLE` environment variable.', context=context)
        if not os.path.isfile(fname):
            time.sleep(0.2)
            continue
        try:
            pf = port_file.PortFile()
            pf.read(fname)
            if not pf.is_valid:
                time.sleep(0.2)
                continue
            self._sock_port = pf.sock_port
        except Exception as e:
            raise ServiceStartPortError(f'Failed to allocate port for wandb service: {e}.')
        return
    raise ServiceStartTimeoutError(f'Timed out waiting for wandb service to start after {self._settings._service_wait} seconds. Try increasing the timeout with the `_service_wait` setting.')