from __future__ import annotations
import sys
from typing import Any
from jupyter_server.extension.application import ExtensionApp, ExtensionAppJinjaMixin
from tornado.ioloop import IOLoop
from .handlers import LabConfig, add_handlers
from .process import Process
Initialize the handlers.