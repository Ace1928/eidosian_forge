from __future__ import annotations
import sys
import time
import warnings
import gevent
from gevent.event import AsyncResult
from gevent.hub import get_hub
import zmq
from zmq import Context as _original_Context
from zmq import Socket as _original_Socket
from .poll import _Poller
class _Socket(_original_Socket):
    """Green version of :class:`zmq.Socket`

    The following methods are overridden:

        * send
        * recv

    To ensure that the ``zmq.NOBLOCK`` flag is set and that sending or receiving
    is deferred to the hub if a ``zmq.EAGAIN`` (retry) error is raised.

    The `__state_changed` method is triggered when the zmq.FD for the socket is
    marked as readable and triggers the necessary read and write events (which
    are waited for in the recv and send methods).

    Some double underscore prefixes are used to minimize pollution of
    :class:`zmq.Socket`'s namespace.
    """
    __in_send_multipart = False
    __in_recv_multipart = False
    __writable = None
    __readable = None
    _state_event = None
    _gevent_bug_timeout = 11.6
    _debug_gevent = False
    _poller_class = _Poller
    _repr_cls = 'zmq.green.Socket'

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.__in_send_multipart = False
        self.__in_recv_multipart = False
        self.__setup_events()

    def __del__(self):
        self.close()

    def close(self, linger=None):
        super().close(linger)
        self.__cleanup_events()

    def __cleanup_events(self):
        if getattr(self, '_state_event', None):
            _stop(self._state_event)
            self._state_event = None
        self.__writable.set()
        self.__readable.set()

    def __setup_events(self):
        self.__readable = AsyncResult()
        self.__writable = AsyncResult()
        self.__readable.set()
        self.__writable.set()
        try:
            self._state_event = get_hub().loop.io(self.getsockopt(zmq.FD), 1)
            self._state_event.start(self.__state_changed)
        except AttributeError:
            from gevent.core import read_event
            self._state_event = read_event(self.getsockopt(zmq.FD), self.__state_changed, persist=True)

    def __state_changed(self, event=None, _evtype=None):
        if self.closed:
            self.__cleanup_events()
            return
        try:
            events = super().getsockopt(zmq.EVENTS)
        except zmq.ZMQError as exc:
            self.__writable.set_exception(exc)
            self.__readable.set_exception(exc)
        else:
            if events & zmq.POLLOUT:
                self.__writable.set()
            if events & zmq.POLLIN:
                self.__readable.set()

    def _wait_write(self):
        assert self.__writable.ready(), 'Only one greenlet can be waiting on this event'
        self.__writable = AsyncResult()
        tic = time.time()
        dt = self._gevent_bug_timeout
        if dt:
            timeout = gevent.Timeout(seconds=dt)
        else:
            timeout = None
        try:
            if timeout:
                timeout.start()
            self.__writable.get(block=True)
        except gevent.Timeout as t:
            if t is not timeout:
                raise
            toc = time.time()
            if self._debug_gevent and timeout and (toc - tic > dt) and self.getsockopt(zmq.EVENTS) & zmq.POLLOUT:
                print('BUG: gevent may have missed a libzmq send event on %i!' % self.FD, file=sys.stderr)
        finally:
            if timeout:
                timeout.close()
            self.__writable.set()

    def _wait_read(self):
        assert self.__readable.ready(), 'Only one greenlet can be waiting on this event'
        self.__readable = AsyncResult()
        tic = time.time()
        dt = self._gevent_bug_timeout
        if dt:
            timeout = gevent.Timeout(seconds=dt)
        else:
            timeout = None
        try:
            if timeout:
                timeout.start()
            self.__readable.get(block=True)
        except gevent.Timeout as t:
            if t is not timeout:
                raise
            toc = time.time()
            if self._debug_gevent and timeout and (toc - tic > dt) and self.getsockopt(zmq.EVENTS) & zmq.POLLIN:
                print('BUG: gevent may have missed a libzmq recv event on %i!' % self.FD, file=sys.stderr)
        finally:
            if timeout:
                timeout.close()
            self.__readable.set()

    def send(self, data, flags=0, copy=True, track=False, **kwargs):
        """send, which will only block current greenlet

        state_changed always fires exactly once (success or fail) at the
        end of this method.
        """
        if flags & zmq.NOBLOCK:
            try:
                msg = super().send(data, flags, copy, track, **kwargs)
            finally:
                if not self.__in_send_multipart:
                    self.__state_changed()
            return msg
        flags |= zmq.NOBLOCK
        while True:
            try:
                msg = super().send(data, flags, copy, track)
            except zmq.ZMQError as e:
                if e.errno != zmq.EAGAIN:
                    if not self.__in_send_multipart:
                        self.__state_changed()
                    raise
            else:
                if not self.__in_send_multipart:
                    self.__state_changed()
                return msg
            self._wait_write()

    def recv(self, flags=0, copy=True, track=False):
        """recv, which will only block current greenlet

        state_changed always fires exactly once (success or fail) at the
        end of this method.
        """
        if flags & zmq.NOBLOCK:
            try:
                msg = super().recv(flags, copy, track)
            finally:
                if not self.__in_recv_multipart:
                    self.__state_changed()
            return msg
        flags |= zmq.NOBLOCK
        while True:
            try:
                msg = super().recv(flags, copy, track)
            except zmq.ZMQError as e:
                if e.errno != zmq.EAGAIN:
                    if not self.__in_recv_multipart:
                        self.__state_changed()
                    raise
            else:
                if not self.__in_recv_multipart:
                    self.__state_changed()
                return msg
            self._wait_read()

    def send_multipart(self, *args, **kwargs):
        """wrap send_multipart to prevent state_changed on each partial send"""
        self.__in_send_multipart = True
        try:
            msg = super().send_multipart(*args, **kwargs)
        finally:
            self.__in_send_multipart = False
            self.__state_changed()
        return msg

    def recv_multipart(self, *args, **kwargs):
        """wrap recv_multipart to prevent state_changed on each partial recv"""
        self.__in_recv_multipart = True
        try:
            msg = super().recv_multipart(*args, **kwargs)
        finally:
            self.__in_recv_multipart = False
            self.__state_changed()
        return msg

    def get(self, opt):
        """trigger state_changed on getsockopt(EVENTS)"""
        if opt in TIMEOS:
            warnings.warn('TIMEO socket options have no effect in zmq.green', UserWarning)
        optval = super().get(opt)
        if opt == zmq.EVENTS:
            self.__state_changed()
        return optval

    def set(self, opt, val):
        """set socket option"""
        if opt in TIMEOS:
            warnings.warn('TIMEO socket options have no effect in zmq.green', UserWarning)
        return super().set(opt, val)