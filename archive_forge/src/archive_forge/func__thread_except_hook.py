import functools
import json
import os
import ssl
import subprocess
import sys
import threading
import time
import traceback
import http.client
import OpenSSL.SSL
import pytest
import requests
import trustme
from .._compat import bton, ntob, ntou
from .._compat import IS_ABOVE_OPENSSL10, IS_CI, IS_PYPY
from .._compat import IS_LINUX, IS_MACOS, IS_WINDOWS
from ..server import HTTPServer, get_ssl_adapter_class
from ..testing import (
from ..wsgi import Gateway_10
def _thread_except_hook(exceptions, args):
    """Append uncaught exception ``args`` in threads to ``exceptions``."""
    if issubclass(args.exc_type, SystemExit):
        return
    exceptions.append((args.exc_type, str(args.exc_value), ''.join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))))