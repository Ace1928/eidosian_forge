from __future__ import annotations
import asyncio
import logging
import signal
import sys
from typing import Any
from gunicorn.arbiter import Arbiter
from gunicorn.workers.base import Worker
from uvicorn.config import Config
from uvicorn.main import Server
def _install_sigquit_handler(self) -> None:
    """Install a SIGQUIT handler on workers.

        - https://github.com/encode/uvicorn/issues/1116
        - https://github.com/benoitc/gunicorn/issues/2604
        """
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGQUIT, self.handle_exit, signal.SIGQUIT, None)