import os
import re
import sys
import time
from io import BytesIO
from typing import Callable, ClassVar, Dict, Iterator, List, Optional, Tuple
from urllib.parse import parse_qs
from wsgiref.simple_server import (
from dulwich import log_utils
from .protocol import ReceivableProtocol
from .repo import BaseRepo, NotGitRepository, Repo
from .server import (
class WSGIRequestHandlerLogger(WSGIRequestHandler):
    """WSGIRequestHandler that uses dulwich's logger for logging exceptions."""

    def log_exception(self, exc_info):
        logger.exception('Exception happened during processing of request', exc_info=exc_info)

    def log_message(self, format, *args):
        logger.info(format, *args)

    def log_error(self, *args):
        logger.error(*args)

    def handle(self):
        """Handle a single HTTP request."""
        self.raw_requestline = self.rfile.readline()
        if not self.parse_request():
            return
        handler = ServerHandlerLogger(self.rfile, self.wfile, self.get_stderr(), self.get_environ())
        handler.request_handler = self
        handler.run(self.server.get_app())