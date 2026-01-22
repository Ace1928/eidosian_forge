import contextlib
import json
import logging
import os
import re
import shlex
import signal
import subprocess
import sys
from importlib import import_module
from multiprocessing import get_context
from multiprocessing.context import SpawnProcess
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union
import anyio
from .filters import DefaultFilter
from .main import Change, FileChange, awatch, watch
class CombinedProcess:

    def __init__(self, p: 'Union[SpawnProcess, subprocess.Popen[bytes]]'):
        self._p = p
        assert self.pid is not None, 'process not yet spawned'

    def stop(self, sigint_timeout: int=5, sigkill_timeout: int=1) -> None:
        os.environ.pop('WATCHFILES_CHANGES', None)
        if self.is_alive():
            logger.debug('stopping process...')
            os.kill(self.pid, signal.SIGINT)
            try:
                self.join(sigint_timeout)
            except subprocess.TimeoutExpired:
                logger.warning('SIGINT timed out after %r seconds', sigint_timeout)
                pass
            if self.exitcode is None:
                logger.warning('process has not terminated, sending SIGKILL')
                os.kill(self.pid, signal.SIGKILL)
                self.join(sigkill_timeout)
            else:
                logger.debug('process stopped')
        else:
            logger.warning('process already dead, exit code: %d', self.exitcode)

    def is_alive(self) -> bool:
        if isinstance(self._p, SpawnProcess):
            return self._p.is_alive()
        else:
            return self._p.poll() is None

    @property
    def pid(self) -> int:
        return self._p.pid

    def join(self, timeout: int) -> None:
        if isinstance(self._p, SpawnProcess):
            self._p.join(timeout)
        else:
            self._p.wait(timeout)

    @property
    def exitcode(self) -> Optional[int]:
        if isinstance(self._p, SpawnProcess):
            return self._p.exitcode
        else:
            return self._p.returncode