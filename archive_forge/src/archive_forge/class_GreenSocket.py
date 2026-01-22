import errno
import os
import socket
import sys
import time
import warnings
import eventlet
from eventlet.hubs import trampoline, notify_opened, IOClosed
from eventlet.support import get_errno
class GreenSocket:
    """
    Green version of socket.socket class, that is intended to be 100%
    API-compatible.

    It also recognizes the keyword parameter, 'set_nonblocking=True'.
    Pass False to indicate that socket is already in non-blocking mode
    to save syscalls.
    """
    fd = None

    def __init__(self, family=socket.AF_INET, *args, **kwargs):
        should_set_nonblocking = kwargs.pop('set_nonblocking', True)
        if isinstance(family, int):
            fd = _original_socket(family, *args, **kwargs)
            notify_opened(fd.fileno())
        else:
            fd = family
        try:
            self._timeout = fd.gettimeout() or socket.getdefaulttimeout()
        except AttributeError:
            self._timeout = socket.getdefaulttimeout()
        if should_set_nonblocking and fd.fileno() != -1:
            set_nonblocking(fd)
        self.fd = fd
        self.act_non_blocking = False
        self.bind = fd.bind
        self.close = fd.close
        self.fileno = fd.fileno
        self.getsockname = fd.getsockname
        self.getsockopt = fd.getsockopt
        self.listen = fd.listen
        self.setsockopt = fd.setsockopt
        self.shutdown = fd.shutdown
        self._closed = False

    @property
    def _sock(self):
        return self

    def _get_io_refs(self):
        return self.fd._io_refs

    def _set_io_refs(self, value):
        self.fd._io_refs = value
    _io_refs = property(_get_io_refs, _set_io_refs)

    def __getattr__(self, name):
        if self.fd is None:
            raise AttributeError(name)
        attr = getattr(self.fd, name)
        setattr(self, name, attr)
        return attr

    def _trampoline(self, fd, read=False, write=False, timeout=None, timeout_exc=None):
        """ We need to trampoline via the event hub.
            We catch any signal back from the hub indicating that the operation we
            were waiting on was associated with a filehandle that's since been
            invalidated.
        """
        if self._closed:
            raise IOClosed()
        try:
            return trampoline(fd, read=read, write=write, timeout=timeout, timeout_exc=timeout_exc, mark_as_closed=self._mark_as_closed)
        except IOClosed:
            self._mark_as_closed()
            raise

    def accept(self):
        if self.act_non_blocking:
            res = self.fd.accept()
            notify_opened(res[0].fileno())
            return res
        fd = self.fd
        _timeout_exc = socket_timeout('timed out')
        while True:
            res = socket_accept(fd)
            if res is not None:
                client, addr = res
                notify_opened(client.fileno())
                set_nonblocking(client)
                return (type(self)(client), addr)
            self._trampoline(fd, read=True, timeout=self.gettimeout(), timeout_exc=_timeout_exc)

    def _mark_as_closed(self):
        """ Mark this socket as being closed """
        self._closed = True

    def __del__(self):
        close = getattr(self, 'close', None)
        if close is not None:
            close()

    def connect(self, address):
        if self.act_non_blocking:
            return self.fd.connect(address)
        fd = self.fd
        _timeout_exc = socket_timeout('timed out')
        if self.gettimeout() is None:
            while not socket_connect(fd, address):
                try:
                    self._trampoline(fd, write=True)
                except IOClosed:
                    raise OSError(errno.EBADFD)
                socket_checkerr(fd)
        else:
            end = time.time() + self.gettimeout()
            while True:
                if socket_connect(fd, address):
                    return
                if time.time() >= end:
                    raise _timeout_exc
                timeout = end - time.time()
                try:
                    self._trampoline(fd, write=True, timeout=timeout, timeout_exc=_timeout_exc)
                except IOClosed:
                    raise OSError(errno.EBADFD)
                socket_checkerr(fd)

    def connect_ex(self, address):
        if self.act_non_blocking:
            return self.fd.connect_ex(address)
        fd = self.fd
        if self.gettimeout() is None:
            while not socket_connect(fd, address):
                try:
                    self._trampoline(fd, write=True)
                    socket_checkerr(fd)
                except OSError as ex:
                    return get_errno(ex)
                except IOClosed:
                    return errno.EBADFD
                return 0
        else:
            end = time.time() + self.gettimeout()
            timeout_exc = socket.timeout(errno.EAGAIN)
            while True:
                try:
                    if socket_connect(fd, address):
                        return 0
                    if time.time() >= end:
                        raise timeout_exc
                    self._trampoline(fd, write=True, timeout=end - time.time(), timeout_exc=timeout_exc)
                    socket_checkerr(fd)
                except OSError as ex:
                    return get_errno(ex)
                except IOClosed:
                    return errno.EBADFD
                return 0

    def dup(self, *args, **kw):
        sock = self.fd.dup(*args, **kw)
        newsock = type(self)(sock, set_nonblocking=False)
        newsock.settimeout(self.gettimeout())
        return newsock

    def makefile(self, *args, **kwargs):
        return _original_socket.makefile(self, *args, **kwargs)

    def makeGreenFile(self, *args, **kw):
        warnings.warn('makeGreenFile has been deprecated, please use makefile instead', DeprecationWarning, stacklevel=2)
        return self.makefile(*args, **kw)

    def _read_trampoline(self):
        self._trampoline(self.fd, read=True, timeout=self.gettimeout(), timeout_exc=socket_timeout('timed out'))

    def _recv_loop(self, recv_meth, empty_val, *args):
        if self.act_non_blocking:
            return recv_meth(*args)
        while True:
            try:
                if not args[0]:
                    self._read_trampoline()
                return recv_meth(*args)
            except OSError as e:
                if get_errno(e) in SOCKET_BLOCKING:
                    pass
                elif get_errno(e) in SOCKET_CLOSED:
                    return empty_val
                else:
                    raise
            try:
                self._read_trampoline()
            except IOClosed as e:
                raise EOFError()

    def recv(self, bufsize, flags=0):
        return self._recv_loop(self.fd.recv, b'', bufsize, flags)

    def recvfrom(self, bufsize, flags=0):
        return self._recv_loop(self.fd.recvfrom, b'', bufsize, flags)

    def recv_into(self, buffer, nbytes=0, flags=0):
        return self._recv_loop(self.fd.recv_into, 0, buffer, nbytes, flags)

    def recvfrom_into(self, buffer, nbytes=0, flags=0):
        return self._recv_loop(self.fd.recvfrom_into, 0, buffer, nbytes, flags)

    def _send_loop(self, send_method, data, *args):
        if self.act_non_blocking:
            return send_method(data, *args)
        _timeout_exc = socket_timeout('timed out')
        while True:
            try:
                return send_method(data, *args)
            except OSError as e:
                eno = get_errno(e)
                if eno == errno.ENOTCONN or eno not in SOCKET_BLOCKING:
                    raise
            try:
                self._trampoline(self.fd, write=True, timeout=self.gettimeout(), timeout_exc=_timeout_exc)
            except IOClosed:
                raise OSError(errno.ECONNRESET, 'Connection closed by another thread')

    def send(self, data, flags=0):
        return self._send_loop(self.fd.send, data, flags)

    def sendto(self, data, *args):
        return self._send_loop(self.fd.sendto, data, *args)

    def sendall(self, data, flags=0):
        tail = self.send(data, flags)
        len_data = len(data)
        while tail < len_data:
            tail += self.send(data[tail:], flags)

    def setblocking(self, flag):
        if flag:
            self.act_non_blocking = False
            self._timeout = None
        else:
            self.act_non_blocking = True
            self._timeout = 0.0

    def settimeout(self, howlong):
        if howlong is None or howlong == _GLOBAL_DEFAULT_TIMEOUT:
            self.setblocking(True)
            return
        try:
            f = howlong.__float__
        except AttributeError:
            raise TypeError('a float is required')
        howlong = f()
        if howlong < 0.0:
            raise ValueError('Timeout value out of range')
        if howlong == 0.0:
            self.act_non_blocking = True
            self._timeout = 0.0
        else:
            self.act_non_blocking = False
            self._timeout = howlong

    def gettimeout(self):
        return self._timeout

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
    if '__pypy__' in sys.builtin_module_names:

        def _reuse(self):
            getattr(self.fd, '_sock', self.fd)._reuse()

        def _drop(self):
            getattr(self.fd, '_sock', self.fd)._drop()