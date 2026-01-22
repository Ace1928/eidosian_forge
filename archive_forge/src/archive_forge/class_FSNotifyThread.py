import linecache
import os
from _pydev_bundle.pydev_imports import _queue
from _pydev_bundle._pydev_saved_modules import time
from _pydev_bundle._pydev_saved_modules import threading
from _pydev_bundle._pydev_saved_modules import socket as socket_module
from _pydevd_bundle.pydevd_constants import (DebugInfoHolder, IS_WINDOWS, IS_JYTHON, IS_WASM,
from _pydev_bundle.pydev_override import overrides
import weakref
from _pydev_bundle._pydev_completer import extract_token_and_qualifier
from _pydevd_bundle._debug_adapter.pydevd_schema import VariablesResponseBody, \
from _pydevd_bundle._debug_adapter import pydevd_base_schema, pydevd_schema
from _pydevd_bundle.pydevd_net_command import NetCommand
from _pydevd_bundle.pydevd_xml import ExceptionOnEvaluate
from _pydevd_bundle.pydevd_constants import ForkSafeLock, NULL
from _pydevd_bundle.pydevd_daemon_thread import PyDBDaemonThread
from _pydevd_bundle.pydevd_thread_lifecycle import pydevd_find_thread_by_id, resume_threads
from _pydevd_bundle.pydevd_dont_trace_files import PYDEV_FILE
import dis
import pydevd_file_utils
import itertools
from urllib.parse import quote_plus, unquote_plus
import pydevconsole
from _pydevd_bundle import pydevd_vars, pydevd_io, pydevd_reload
from _pydevd_bundle import pydevd_bytecode_utils
from _pydevd_bundle import pydevd_xml
from _pydevd_bundle import pydevd_vm_type
import sys
import traceback
from _pydevd_bundle.pydevd_utils import quote_smart as quote, compare_object_attrs_key, \
from _pydev_bundle import pydev_log, fsnotify
from _pydev_bundle.pydev_log import exception as pydev_log_exception
from _pydev_bundle import _pydev_completer
from pydevd_tracing import get_exception_traceback_str
from _pydevd_bundle import pydevd_console
from _pydev_bundle.pydev_monkey import disable_trace_thread_modules, enable_trace_thread_modules
from io import StringIO
from _pydevd_bundle.pydevd_comm_constants import *  # @UnusedWildImport
class FSNotifyThread(PyDBDaemonThread):

    def __init__(self, py_db, api, watch_dirs):
        PyDBDaemonThread.__init__(self, py_db)
        self.api = api
        self.name = 'pydevd.FSNotifyThread'
        self.watcher = fsnotify.Watcher()
        self.watch_dirs = watch_dirs

    @overrides(PyDBDaemonThread._on_run)
    def _on_run(self):
        try:
            pydev_log.info('Watching directories for code reload:\n---\n%s\n---' % '\n'.join(sorted(self.watch_dirs)))
            self.watcher.set_tracked_paths(self.watch_dirs)
            while not self._kill_received:
                for change_enum, change_path in self.watcher.iter_changes():
                    if change_enum == fsnotify.Change.modified:
                        pydev_log.info('Modified: %s', change_path)
                        self.api.request_reload_code(self.py_db, -1, None, change_path)
                    else:
                        pydev_log.info('Ignored (add or remove) change in: %s', change_path)
        except:
            pydev_log.exception('Error when waiting for filesystem changes in FSNotifyThread.')

    @overrides(PyDBDaemonThread.do_kill_pydev_thread)
    def do_kill_pydev_thread(self):
        self.watcher.dispose()
        PyDBDaemonThread.do_kill_pydev_thread(self)