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
class wait_for_client:

    def __call__(self):
        ensure_logging()
        log.debug('wait_for_client()')
        pydb = get_global_debugger()
        if pydb is None:
            raise RuntimeError('listen() or connect() must be called first')
        cancel_event = threading.Event()
        self.cancel = cancel_event.set
        pydevd._wait_for_attach(cancel=cancel_event)

    @staticmethod
    def cancel():
        raise RuntimeError('wait_for_client() must be called first')