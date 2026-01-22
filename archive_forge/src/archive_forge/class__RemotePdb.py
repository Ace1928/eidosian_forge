import errno
import inspect
import json
import logging
import os
import re
import select
import socket
import sys
import time
import traceback
import uuid
from pdb import Pdb
from typing import Callable
import setproctitle
import ray
from ray._private import ray_constants
from ray.experimental.internal_kv import _internal_kv_del, _internal_kv_put
from ray.util.annotations import DeveloperAPI
class _RemotePdb(Pdb):
    """
    This will run pdb as a ephemeral telnet service. Once you connect no one
    else can connect. On construction this object will block execution till a
    client has connected.
    Based on https://github.com/tamentis/rpdb I think ...
    To use this::
        RemotePdb(host="0.0.0.0", port=4444).set_trace()
    Then run: telnet 127.0.0.1 4444
    """
    active_instance = None

    def __init__(self, breakpoint_uuid, host, port, ip_address, patch_stdstreams=False, quiet=False):
        self._breakpoint_uuid = breakpoint_uuid
        self._quiet = quiet
        self._patch_stdstreams = patch_stdstreams
        self._listen_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listen_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
        self._listen_socket.bind((host, port))
        self._ip_address = ip_address

    def listen(self):
        if not self._quiet:
            _cry("RemotePdb session open at %s:%s, use 'ray debug' to connect..." % (self._ip_address, self._listen_socket.getsockname()[1]))
        self._listen_socket.listen(1)
        connection, address = self._listen_socket.accept()
        if not self._quiet:
            _cry('RemotePdb accepted connection from %s.' % repr(address))
        self.handle = _LF2CRLF_FileWrapper(connection)
        Pdb.__init__(self, completekey='tab', stdin=self.handle, stdout=self.handle, skip=['ray.*'])
        self.backup = []
        if self._patch_stdstreams:
            for name in ('stderr', 'stdout', '__stderr__', '__stdout__', 'stdin', '__stdin__'):
                self.backup.append((name, getattr(sys, name)))
                setattr(sys, name, self.handle)
        _RemotePdb.active_instance = self

    def __restore(self):
        if self.backup and (not self._quiet):
            _cry('Restoring streams: %s ...' % self.backup)
        for name, fh in self.backup:
            setattr(sys, name, fh)
        self.handle.close()
        _RemotePdb.active_instance = None

    def do_quit(self, arg):
        self.__restore()
        return Pdb.do_quit(self, arg)
    do_q = do_exit = do_quit

    def do_continue(self, arg):
        self.__restore()
        self.handle.connection.close()
        return Pdb.do_continue(self, arg)
    do_c = do_cont = do_continue

    def set_trace(self, frame=None):
        if frame is None:
            frame = sys._getframe().f_back
        try:
            Pdb.set_trace(self, frame)
        except IOError as exc:
            if exc.errno != errno.ECONNRESET:
                raise

    def post_mortem(self, traceback=None):
        try:
            t = sys.exc_info()[2]
            self.reset()
            Pdb.interaction(self, None, t)
        except IOError as exc:
            if exc.errno != errno.ECONNRESET:
                raise

    def do_remote(self, arg):
        """remote
        Skip into the next remote call.
        """
        ray._private.worker.global_worker.debugger_breakpoint = self._breakpoint_uuid
        data = json.dumps({'job_id': ray.get_runtime_context().get_job_id()})
        _internal_kv_put('RAY_PDB_CONTINUE_{}'.format(self._breakpoint_uuid), data, namespace=ray_constants.KV_NAMESPACE_PDB)
        self.__restore()
        self.handle.connection.close()
        return Pdb.do_continue(self, arg)

    def do_get(self, arg):
        """get
        Skip to where the current task returns to.
        """
        ray._private.worker.global_worker.debugger_get_breakpoint = self._breakpoint_uuid
        self.__restore()
        self.handle.connection.close()
        return Pdb.do_continue(self, arg)