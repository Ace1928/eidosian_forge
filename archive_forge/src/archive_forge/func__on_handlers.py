import asyncio
import atexit
import os
import string
import subprocess
from datetime import datetime, timezone
from tornado.ioloop import IOLoop
from tornado.queues import Queue
from tornado.websocket import WebSocketHandler
from traitlets import Bunch, Instance, Set, Unicode, UseEnum, observe
from traitlets.config import LoggingConfigurable
from . import stdio
from .schema import LANGUAGE_SERVER_SPEC
from .specs.utils import censored_spec
from .trait_types import Schema
from .types import SessionStatus
@observe('handlers')
def _on_handlers(self, change: Bunch):
    """re-initialize if someone starts listening, or stop if nobody is"""
    if change['new'] and (not self.process):
        self.initialize()
    elif not change['new'] and self.process:
        self.stop()