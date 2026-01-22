import asyncio
import inspect
import logging
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from os import path as osp
from jupyter_server.serverapp import aliases, flags
from jupyter_server.utils import pathname2url, urljoin
from tornado.ioloop import IOLoop
from tornado.iostream import StreamClosedError
from tornado.websocket import WebSocketClosedError
from traitlets import Bool, Unicode
from .labapp import LabApp, get_app_dir
from .tests.test_app import TestEnv
class LogErrorHandler(logging.Handler):
    """A handler that exits with 1 on a logged error."""

    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.errored = False

    def filter(self, record):
        if hasattr(record, 'exc_info') and record.exc_info is not None and isinstance(record.exc_info[1], (StreamClosedError, WebSocketClosedError)):
            return
        return super().filter(record)

    def emit(self, record):
        print(record.msg, file=sys.stderr)
        self.errored = True