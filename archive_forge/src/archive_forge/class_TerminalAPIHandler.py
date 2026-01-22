from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from jupyter_server.auth.decorator import authorized
from jupyter_server.base.handlers import APIHandler
from tornado import web
from .base import TerminalsMixin
class TerminalAPIHandler(APIHandler):
    """The base terminal handler."""
    auth_resource = AUTH_RESOURCE