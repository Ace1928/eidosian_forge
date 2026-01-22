import sys
import threading
import signal
import array
import queue
import time
import types
import os
from os import getpid
from traceback import format_exc
from . import connection
from .context import reduction, get_spawning_popen, ProcessError
from . import pool
from . import process
from . import util
from . import get_context
class SharedMemoryManager(BaseManager):
    """Like SyncManager but uses SharedMemoryServer instead of Server.

        It provides methods for creating and returning SharedMemory instances
        and for creating a list-like object (ShareableList) backed by shared
        memory.  It also provides methods that create and return Proxy Objects
        that support synchronization across processes (i.e. multi-process-safe
        locks and semaphores).
        """
    _Server = SharedMemoryServer

    def __init__(self, *args, **kwargs):
        if os.name == 'posix':
            from . import resource_tracker
            resource_tracker.ensure_running()
        BaseManager.__init__(self, *args, **kwargs)
        util.debug(f'{self.__class__.__name__} created by pid {getpid()}')

    def __del__(self):
        util.debug(f'{self.__class__.__name__}.__del__ by pid {getpid()}')
        pass

    def get_server(self):
        """Better than monkeypatching for now; merge into Server ultimately"""
        if self._state.value != State.INITIAL:
            if self._state.value == State.STARTED:
                raise ProcessError('Already started SharedMemoryServer')
            elif self._state.value == State.SHUTDOWN:
                raise ProcessError('SharedMemoryManager has shut down')
            else:
                raise ProcessError('Unknown state {!r}'.format(self._state.value))
        return self._Server(self._registry, self._address, self._authkey, self._serializer)

    def SharedMemory(self, size):
        """Returns a new SharedMemory instance with the specified size in
            bytes, to be tracked by the manager."""
        with self._Client(self._address, authkey=self._authkey) as conn:
            sms = shared_memory.SharedMemory(None, create=True, size=size)
            try:
                dispatch(conn, None, 'track_segment', (sms.name,))
            except BaseException as e:
                sms.unlink()
                raise e
        return sms

    def ShareableList(self, sequence):
        """Returns a new ShareableList instance populated with the values
            from the input sequence, to be tracked by the manager."""
        with self._Client(self._address, authkey=self._authkey) as conn:
            sl = shared_memory.ShareableList(sequence)
            try:
                dispatch(conn, None, 'track_segment', (sl.shm.name,))
            except BaseException as e:
                sl.shm.unlink()
                raise e
        return sl