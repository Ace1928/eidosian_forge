import sys
import bisect
import types
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle import pydevd_utils, pydevd_source_mapping
from _pydevd_bundle.pydevd_additional_thread_info import set_additional_thread_info
from _pydevd_bundle.pydevd_comm import (InternalGetThreadStack, internal_get_completions,
from _pydevd_bundle.pydevd_comm_constants import (CMD_THREAD_SUSPEND, file_system_encoding,
from _pydevd_bundle.pydevd_constants import (get_current_thread_id, set_protocol, get_protocol,
from _pydevd_bundle.pydevd_net_command_factory_json import NetCommandFactoryJson
from _pydevd_bundle.pydevd_net_command_factory_xml import NetCommandFactory
import pydevd_file_utils
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_breakpoints import LineBreakpoint
from pydevd_tracing import get_exception_traceback_str
import os
import subprocess
import ctypes
from _pydevd_bundle.pydevd_collect_bytecode_info import code_to_bytecode_representation
import itertools
import linecache
from _pydevd_bundle.pydevd_utils import DAPGrouper, interrupt_main_thread
from _pydevd_bundle.pydevd_daemon_thread import run_as_pydevd_daemon_thread
from _pydevd_bundle.pydevd_thread_lifecycle import pydevd_find_thread_by_id, resume_threads
import tokenize
def _terminate_child_processes_linux_and_mac(self, dont_terminate_child_pids):
    this_pid = os.getpid()

    def list_children_and_stop_forking(initial_pid, stop=True):
        children_pids = []
        if stop:
            self._call(['kill', '-STOP', str(initial_pid)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        list_popen = self._popen(['pgrep', '-P', str(initial_pid)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if list_popen is not None:
            stdout, _ = list_popen.communicate()
            for line in stdout.splitlines():
                line = line.decode('ascii').strip()
                if line:
                    pid = str(line)
                    if pid in dont_terminate_child_pids:
                        continue
                    children_pids.append(pid)
                    children_pids.extend(list_children_and_stop_forking(pid))
        return children_pids
    previously_found = set()
    for _ in range(50):
        children_pids = list_children_and_stop_forking(this_pid, stop=False)
        found_new = False
        for pid in children_pids:
            if pid not in previously_found:
                found_new = True
                previously_found.add(pid)
                self._call(['kill', '-KILL', str(pid)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if not found_new:
            break