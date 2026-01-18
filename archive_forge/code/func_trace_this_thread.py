import codecs
import os
import pydevd
import socket
import sys
import threading
import debugpy
from debugpy import adapter
from debugpy.common import json, log, sockets
from _pydevd_bundle.pydevd_constants import get_global_debugger
from pydevd_file_utils import absolute_path
from debugpy.common.util import hide_debugpy_internals
def trace_this_thread(should_trace):
    ensure_logging()
    log.debug('trace_this_thread({0!r})', should_trace)
    pydb = get_global_debugger()
    if should_trace:
        pydb.enable_tracing()
    else:
        pydb.disable_tracing()