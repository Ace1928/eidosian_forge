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
class ResultHandler(_pool.ResultHandler):
    """Handles messages from the pool processes."""

    def __init__(self, *args, **kwargs):
        self.fileno_to_outq = kwargs.pop('fileno_to_outq')
        self.on_process_alive = kwargs.pop('on_process_alive')
        super().__init__(*args, **kwargs)
        self.state_handlers[WORKER_UP] = self.on_process_alive

    def _recv_message(self, add_reader, fd, callback, __read__=__read__, readcanbuf=readcanbuf, BytesIO=BytesIO, unpack_from=unpack_from, load=_pickle.load):
        Hr = Br = 0
        if readcanbuf:
            buf = bytearray(4)
            bufv = memoryview(buf)
        else:
            buf = bufv = BytesIO()
        while Hr < 4:
            try:
                n = __read__(fd, bufv[Hr:] if readcanbuf else bufv, 4 - Hr)
            except OSError as exc:
                if exc.errno not in UNAVAIL:
                    raise
                yield
            else:
                if n == 0:
                    raise OSError('End of file during message') if Hr else EOFError()
                Hr += n
        body_size, = unpack_from('>i', bufv)
        if readcanbuf:
            buf = bytearray(body_size)
            bufv = memoryview(buf)
        else:
            buf = bufv = BytesIO()
        while Br < body_size:
            try:
                n = __read__(fd, bufv[Br:] if readcanbuf else bufv, body_size - Br)
            except OSError as exc:
                if exc.errno not in UNAVAIL:
                    raise
                yield
            else:
                if n == 0:
                    raise OSError('End of file during message') if Br else EOFError()
                Br += n
        add_reader(fd, self.handle_event, fd)
        if readcanbuf:
            message = load(BytesIO(bufv))
        else:
            bufv.seek(0)
            message = load(bufv)
        if message:
            callback(message)

    def _make_process_result(self, hub):
        """Coroutine reading messages from the pool processes."""
        fileno_to_outq = self.fileno_to_outq
        on_state_change = self.on_state_change
        add_reader = hub.add_reader
        remove_reader = hub.remove_reader
        recv_message = self._recv_message

        def on_result_readable(fileno):
            try:
                fileno_to_outq[fileno]
            except KeyError:
                return remove_reader(fileno)
            it = recv_message(add_reader, fileno, on_state_change)
            try:
                next(it)
            except StopIteration:
                pass
            except (OSError, EOFError):
                remove_reader(fileno)
            else:
                add_reader(fileno, it)
        return on_result_readable

    def register_with_event_loop(self, hub):
        self.handle_event = self._make_process_result(hub)

    def handle_event(self, *args):
        raise RuntimeError('Not registered with event loop')

    def on_stop_not_started(self):
        cache = self.cache
        check_timeouts = self.check_timeouts
        fileno_to_outq = self.fileno_to_outq
        on_state_change = self.on_state_change
        join_exited_workers = self.join_exited_workers
        outqueues = set(fileno_to_outq)
        while cache and outqueues and (self._state != TERMINATE):
            if check_timeouts is not None:
                check_timeouts()
            pending_remove_fd = set()
            for fd in outqueues:
                iterate_file_descriptors_safely([fd], self.fileno_to_outq, self._flush_outqueue, pending_remove_fd.add, fileno_to_outq, on_state_change)
                try:
                    join_exited_workers(shutdown=True)
                except WorkersJoined:
                    debug('result handler: all workers terminated')
                    return
            outqueues.difference_update(pending_remove_fd)

    def _flush_outqueue(self, fd, remove, process_index, on_state_change):
        try:
            proc = process_index[fd]
        except KeyError:
            return remove(fd)
        reader = proc.outq._reader
        try:
            setblocking(reader, 1)
        except OSError:
            return remove(fd)
        try:
            if reader.poll(0):
                task = reader.recv()
            else:
                task = None
                sleep(0.5)
        except (OSError, EOFError):
            return remove(fd)
        else:
            if task:
                on_state_change(task)
        finally:
            try:
                setblocking(reader, 0)
            except OSError:
                return remove(fd)