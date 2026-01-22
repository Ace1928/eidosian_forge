import errno
import gc
import inspect
import os
import select
import time
from collections import Counter, deque, namedtuple
from io import BytesIO
from numbers import Integral
from pickle import HIGHEST_PROTOCOL
from struct import pack, unpack, unpack_from
from time import sleep
from weakref import WeakValueDictionary, ref
from billiard import pool as _pool
from billiard.compat import isblocking, setblocking
from billiard.pool import ACK, NACK, RUN, TERMINATE, WorkersJoined
from billiard.queues import _SimpleQueue
from kombu.asynchronous import ERR, WRITE
from kombu.serialization import pickle as _pickle
from kombu.utils.eventio import SELECT_BAD_FD
from kombu.utils.functional import fxrange
from vine import promise
from celery.signals import worker_before_create_process
from celery.utils.functional import noop
from celery.utils.log import get_logger
from celery.worker import state as worker_state
def _create_process_handlers(self, hub):
    """Create handlers called on process up/down, etc."""
    add_reader, remove_reader, remove_writer = (hub.add_reader, hub.remove_reader, hub.remove_writer)
    cache = self._cache
    all_inqueues = self._all_inqueues
    fileno_to_inq = self._fileno_to_inq
    fileno_to_outq = self._fileno_to_outq
    fileno_to_synq = self._fileno_to_synq
    busy_workers = self._busy_workers
    handle_result_event = self.handle_result_event
    process_flush_queues = self.process_flush_queues
    waiting_to_start = self._waiting_to_start

    def verify_process_alive(proc):
        proc = proc()
        if proc is not None and proc._is_alive() and (proc in waiting_to_start):
            assert proc.outqR_fd in fileno_to_outq
            assert fileno_to_outq[proc.outqR_fd] is proc
            assert proc.outqR_fd in hub.readers
            error('Timed out waiting for UP message from %r', proc)
            os.kill(proc.pid, 9)

    def on_process_up(proc):
        """Called when a process has started."""
        infd = proc.inqW_fd
        for job in cache.values():
            if job._write_to and job._write_to.inqW_fd == infd:
                job._write_to = proc
            if job._scheduled_for and job._scheduled_for.inqW_fd == infd:
                job._scheduled_for = proc
        fileno_to_outq[proc.outqR_fd] = proc
        self._track_child_process(proc, hub)
        assert not isblocking(proc.outq._reader)
        add_reader(proc.outqR_fd, handle_result_event, proc.outqR_fd)
        waiting_to_start.add(proc)
        hub.call_later(self._proc_alive_timeout, verify_process_alive, ref(proc))
    self.on_process_up = on_process_up

    def _remove_from_index(obj, proc, index, remove_fun, callback=None):
        try:
            fd = obj.fileno()
        except OSError:
            return
        try:
            if index[fd] is proc:
                index.pop(fd, None)
        except KeyError:
            pass
        else:
            remove_fun(fd)
            if callback is not None:
                callback(fd)
        return fd

    def on_process_down(proc):
        """Called when a worker process exits."""
        if getattr(proc, 'dead', None):
            return
        process_flush_queues(proc)
        _remove_from_index(proc.outq._reader, proc, fileno_to_outq, remove_reader)
        if proc.synq:
            _remove_from_index(proc.synq._writer, proc, fileno_to_synq, remove_writer)
        inq = _remove_from_index(proc.inq._writer, proc, fileno_to_inq, remove_writer, callback=all_inqueues.discard)
        if inq:
            busy_workers.discard(inq)
        self._untrack_child_process(proc, hub)
        waiting_to_start.discard(proc)
        self._active_writes.discard(proc.inqW_fd)
        remove_writer(proc.inq._writer)
        remove_reader(proc.outq._reader)
        if proc.synqR_fd:
            remove_reader(proc.synq._reader)
        if proc.synqW_fd:
            self._active_writes.discard(proc.synqW_fd)
            remove_reader(proc.synq._writer)
    self.on_process_down = on_process_down