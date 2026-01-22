import errno
import os
import selectors
import signal
import socket
import struct
import sys
import threading
import warnings
from . import connection
from . import process
from .context import reduction
from . import resource_tracker
from . import spawn
from . import util
class ForkServer(object):

    def __init__(self):
        self._forkserver_address = None
        self._forkserver_alive_fd = None
        self._forkserver_pid = None
        self._inherited_fds = None
        self._lock = threading.Lock()
        self._preload_modules = ['__main__']

    def _stop(self):
        with self._lock:
            self._stop_unlocked()

    def _stop_unlocked(self):
        if self._forkserver_pid is None:
            return
        os.close(self._forkserver_alive_fd)
        self._forkserver_alive_fd = None
        os.waitpid(self._forkserver_pid, 0)
        self._forkserver_pid = None
        if not util.is_abstract_socket_namespace(self._forkserver_address):
            os.unlink(self._forkserver_address)
        self._forkserver_address = None

    def set_forkserver_preload(self, modules_names):
        """Set list of module names to try to load in forkserver process."""
        if not all((type(mod) is str for mod in self._preload_modules)):
            raise TypeError('module_names must be a list of strings')
        self._preload_modules = modules_names

    def get_inherited_fds(self):
        """Return list of fds inherited from parent process.

        This returns None if the current process was not started by fork
        server.
        """
        return self._inherited_fds

    def connect_to_new_process(self, fds):
        """Request forkserver to create a child process.

        Returns a pair of fds (status_r, data_w).  The calling process can read
        the child process's pid and (eventually) its returncode from status_r.
        The calling process should write to data_w the pickled preparation and
        process data.
        """
        self.ensure_running()
        if len(fds) + 4 >= MAXFDS_TO_SEND:
            raise ValueError('too many fds')
        with socket.socket(socket.AF_UNIX) as client:
            client.connect(self._forkserver_address)
            parent_r, child_w = os.pipe()
            child_r, parent_w = os.pipe()
            allfds = [child_r, child_w, self._forkserver_alive_fd, resource_tracker.getfd()]
            allfds += fds
            try:
                reduction.sendfds(client, allfds)
                return (parent_r, parent_w)
            except:
                os.close(parent_r)
                os.close(parent_w)
                raise
            finally:
                os.close(child_r)
                os.close(child_w)

    def ensure_running(self):
        """Make sure that a fork server is running.

        This can be called from any process.  Note that usually a child
        process will just reuse the forkserver started by its parent, so
        ensure_running() will do nothing.
        """
        with self._lock:
            resource_tracker.ensure_running()
            if self._forkserver_pid is not None:
                pid, status = os.waitpid(self._forkserver_pid, os.WNOHANG)
                if not pid:
                    return
                os.close(self._forkserver_alive_fd)
                self._forkserver_address = None
                self._forkserver_alive_fd = None
                self._forkserver_pid = None
            cmd = 'from multiprocess.forkserver import main; ' + 'main(%d, %d, %r, **%r)'
            if self._preload_modules:
                desired_keys = {'main_path', 'sys_path'}
                data = spawn.get_preparation_data('ignore')
                data = {x: y for x, y in data.items() if x in desired_keys}
            else:
                data = {}
            with socket.socket(socket.AF_UNIX) as listener:
                address = connection.arbitrary_address('AF_UNIX')
                listener.bind(address)
                if not util.is_abstract_socket_namespace(address):
                    os.chmod(address, 384)
                listener.listen()
                alive_r, alive_w = os.pipe()
                try:
                    fds_to_pass = [listener.fileno(), alive_r]
                    cmd %= (listener.fileno(), alive_r, self._preload_modules, data)
                    exe = spawn.get_executable()
                    args = [exe] + util._args_from_interpreter_flags()
                    args += ['-c', cmd]
                    pid = util.spawnv_passfds(exe, args, fds_to_pass)
                except:
                    os.close(alive_w)
                    raise
                finally:
                    os.close(alive_r)
                self._forkserver_address = address
                self._forkserver_alive_fd = alive_w
                self._forkserver_pid = pid