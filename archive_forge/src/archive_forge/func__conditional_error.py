import os
import io
import re
import email.utils
import socket
import sys
import time
import traceback as traceback_
import logging
import platform
import queue
import contextlib
import threading
import urllib.parse
from functools import lru_cache
from . import connections, errors, __version__
from ._compat import bton
from ._compat import IS_PPC
from .workers import threadpool
from .makefile import MakeFile, StreamWriter
def _conditional_error(self, req, response):
    """Respond with an error.

        Don't bother writing if a response
        has already started being written.
        """
    if not req or req.sent_headers:
        return
    try:
        req.simple_response(response)
    except errors.FatalSSLAlert:
        pass
    except errors.NoSSLError:
        self._handle_no_ssl(req)