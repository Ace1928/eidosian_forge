from __future__ import annotations
import time
import warnings
from threading import Event
from weakref import ref
import cython as C
from cython import (
from cython.cimports.cpython import (
from cython.cimports.libc.errno import EAGAIN, EINTR, ENAMETOOLONG, ENOENT, ENOTSOCK
from cython.cimports.libc.stdint import uint32_t
from cython.cimports.libc.stdio import fprintf
from cython.cimports.libc.stdio import stderr as cstderr
from cython.cimports.libc.stdlib import free, malloc
from cython.cimports.libc.string import memcpy
from cython.cimports.zmq.backend.cython._externs import (
from cython.cimports.zmq.backend.cython.libzmq import (
from cython.cimports.zmq.backend.cython.libzmq import zmq_errno as _zmq_errno
from cython.cimports.zmq.backend.cython.libzmq import zmq_poll as zmq_poll_c
from cython.cimports.zmq.utils.buffers import asbuffer_r
import zmq
from zmq.constants import SocketOption, _OptType
from zmq.error import InterruptedSystemCall, ZMQError, _check_version
@cclass
class Socket:
    """
    A 0MQ socket.

    These objects will generally be constructed via the socket() method of a Context object.

    Note: 0MQ Sockets are *not* threadsafe. **DO NOT** share them across threads.

    Parameters
    ----------
    context : Context
        The 0MQ Context this Socket belongs to.
    socket_type : int
        The socket type, which can be any of the 0MQ socket types:
        REQ, REP, PUB, SUB, PAIR, DEALER, ROUTER, PULL, PUSH, XPUB, XSUB.

    See Also
    --------
    .Context.socket : method for creating a socket bound to a Context.
    """

    def __init__(self, context=None, socket_type: C.int=-1, shadow: size_t=0, copy_threshold=None):
        self.handle = NULL
        self._pid = 0
        self._shadow = False
        self.context = None
        if copy_threshold is None:
            copy_threshold = zmq.COPY_THRESHOLD
        self.copy_threshold = copy_threshold
        self.handle = NULL
        self.context = context
        if shadow:
            self._shadow = True
            self.handle = cast(p_void, shadow)
        else:
            if context is None:
                raise TypeError('context must be specified')
            if socket_type < 0:
                raise TypeError('socket_type must be specified')
            self._shadow = False
            self.handle = zmq_socket(self.context.handle, socket_type)
        if self.handle == NULL:
            raise ZMQError()
        self._closed = False
        self._pid = getpid()

    @property
    def underlying(self):
        """The address of the underlying libzmq socket"""
        return cast(size_t, self.handle)

    @property
    def closed(self):
        """Whether the socket is closed"""
        return _check_closed_deep(self)

    def close(self, linger=None):
        """
        Close the socket.

        If linger is specified, LINGER sockopt will be set prior to closing.

        This can be called to close the socket by hand. If this is not
        called, the socket will automatically be closed when it is
        garbage collected.
        """
        rc: C.int = 0
        linger_c: C.int
        setlinger: bint = False
        if linger is not None:
            linger_c = linger
            setlinger = True
        if self.handle != NULL and (not self._closed) and (getpid() == self._pid):
            if setlinger:
                zmq_setsockopt(self.handle, ZMQ_LINGER, address(linger_c), sizeof(int))
            rc = zmq_close(self.handle)
            if rc < 0 and zmq_errno() != ENOTSOCK:
                _check_rc(rc)
            self._closed = True
            self.handle = NULL

    def set(self, option: C.int, optval):
        """
        Set socket options.

        See the 0MQ API documentation for details on specific options.

        Parameters
        ----------
        option : int
            The option to set.  Available values will depend on your
            version of libzmq.  Examples include::

                zmq.SUBSCRIBE, UNSUBSCRIBE, IDENTITY, HWM, LINGER, FD

        optval : int or bytes
            The value of the option to set.

        Notes
        -----
        .. warning::

            All options other than zmq.SUBSCRIBE, zmq.UNSUBSCRIBE and
            zmq.LINGER only take effect for subsequent socket bind/connects.
        """
        optval_int64_c: int64_t
        optval_int_c: C.int
        optval_c: p_char
        sz: Py_ssize_t
        _check_closed(self)
        if isinstance(optval, str):
            raise TypeError('unicode not allowed, use setsockopt_string')
        try:
            sopt = SocketOption(option)
        except ValueError:
            opt_type = _OptType.int
        else:
            opt_type = sopt._opt_type
        if opt_type == _OptType.bytes:
            if not isinstance(optval, bytes):
                raise TypeError('expected bytes, got: %r' % optval)
            optval_c = PyBytes_AsString(optval)
            sz = PyBytes_Size(optval)
            _setsockopt(self.handle, option, optval_c, sz)
        elif opt_type == _OptType.int64:
            if not isinstance(optval, int):
                raise TypeError('expected int, got: %r' % optval)
            optval_int64_c = optval
            _setsockopt(self.handle, option, address(optval_int64_c), sizeof(int64_t))
        else:
            if not isinstance(optval, int):
                raise TypeError('expected int, got: %r' % optval)
            optval_int_c = optval
            _setsockopt(self.handle, option, address(optval_int_c), sizeof(int))

    def get(self, option: C.int):
        """
        Get the value of a socket option.

        See the 0MQ API documentation for details on specific options.

        Parameters
        ----------
        option : int
            The option to get.  Available values will depend on your
            version of libzmq.  Examples include::

                zmq.IDENTITY, HWM, LINGER, FD, EVENTS

        Returns
        -------
        optval : int or bytes
            The value of the option as a bytestring or int.
        """
        optval_int64_c = declare(int64_t)
        optval_int_c = declare(C.int)
        optval_fd_c = declare(fd_t)
        identity_str_c = declare(char[255])
        sz: size_t
        _check_closed(self)
        try:
            sopt = SocketOption(option)
        except ValueError:
            opt_type = _OptType.int
        else:
            opt_type = sopt._opt_type
        if opt_type == _OptType.bytes:
            sz = 255
            _getsockopt(self.handle, option, cast(p_void, identity_str_c), address(sz))
            if option != ZMQ_IDENTITY and sz > 0 and (cast(p_char, identity_str_c)[sz - 1] == b'\x00'):
                sz -= 1
            result = PyBytes_FromStringAndSize(cast(p_char, identity_str_c), sz)
        elif opt_type == _OptType.int64:
            sz = sizeof(int64_t)
            _getsockopt(self.handle, option, cast(p_void, address(optval_int64_c)), address(sz))
            result = optval_int64_c
        elif opt_type == _OptType.fd:
            sz = sizeof(fd_t)
            _getsockopt(self.handle, option, cast(p_void, address(optval_fd_c)), address(sz))
            result = optval_fd_c
        else:
            sz = sizeof(int)
            _getsockopt(self.handle, option, cast(p_void, address(optval_int_c)), address(sz))
            result = optval_int_c
        return result

    def bind(self, addr):
        """
        Bind the socket to an address.

        This causes the socket to listen on a network port. Sockets on the
        other side of this connection will use ``Socket.connect(addr)`` to
        connect to this socket.

        Parameters
        ----------
        addr : str
            The address string. This has the form 'protocol://interface:port',
            for example 'tcp://127.0.0.1:5555'. Protocols supported include
            tcp, udp, pgm, epgm, inproc and ipc. If the address is unicode, it is
            encoded to utf-8 first.
        """
        rc: C.int
        c_addr: p_char
        _check_closed(self)
        addr_b = addr
        if isinstance(addr, str):
            addr_b = addr.encode('utf-8')
        elif isinstance(addr_b, bytes):
            addr = addr_b.decode('utf-8')
        if not isinstance(addr_b, bytes):
            raise TypeError('expected str, got: %r' % addr)
        c_addr = addr_b
        rc = zmq_bind(self.handle, c_addr)
        if rc != 0:
            if IPC_PATH_MAX_LEN and zmq_errno() == ENAMETOOLONG:
                path = addr.split('://', 1)[-1]
                msg = f'ipc path "{path}" is longer than {IPC_PATH_MAX_LEN} characters (sizeof(sockaddr_un.sun_path)). zmq.IPC_PATH_MAX_LEN constant can be used to check addr length (if it is defined).'
                raise ZMQError(msg=msg)
            elif zmq_errno() == ENOENT:
                path = addr.split('://', 1)[-1]
                msg = f'No such file or directory for ipc path "{path}".'
                raise ZMQError(msg=msg)
        while True:
            try:
                _check_rc(rc)
            except InterruptedSystemCall:
                rc = zmq_bind(self.handle, c_addr)
                continue
            else:
                break

    def connect(self, addr):
        """
        Connect to a remote 0MQ socket.

        Parameters
        ----------
        addr : str
            The address string. This has the form 'protocol://interface:port',
            for example 'tcp://127.0.0.1:5555'. Protocols supported are
            tcp, udp, pgm, inproc and ipc. If the address is unicode, it is
            encoded to utf-8 first.
        """
        rc: C.int
        c_addr: p_char
        _check_closed(self)
        if isinstance(addr, str):
            addr = addr.encode('utf-8')
        if not isinstance(addr, bytes):
            raise TypeError('expected str, got: %r' % addr)
        c_addr = addr
        while True:
            try:
                rc = zmq_connect(self.handle, c_addr)
                _check_rc(rc)
            except InterruptedSystemCall:
                continue
            else:
                break

    def unbind(self, addr):
        """
        Unbind from an address (undoes a call to bind).

        .. versionadded:: libzmq-3.2
        .. versionadded:: 13.0

        Parameters
        ----------
        addr : str
            The address string. This has the form 'protocol://interface:port',
            for example 'tcp://127.0.0.1:5555'. Protocols supported are
            tcp, udp, pgm, inproc and ipc. If the address is unicode, it is
            encoded to utf-8 first.
        """
        rc: C.int
        c_addr: p_char
        _check_version((3, 2), 'unbind')
        _check_closed(self)
        if isinstance(addr, str):
            addr = addr.encode('utf-8')
        if not isinstance(addr, bytes):
            raise TypeError('expected str, got: %r' % addr)
        c_addr = addr
        rc = zmq_unbind(self.handle, c_addr)
        if rc != 0:
            raise ZMQError()

    def disconnect(self, addr):
        """
        Disconnect from a remote 0MQ socket (undoes a call to connect).

        .. versionadded:: libzmq-3.2
        .. versionadded:: 13.0

        Parameters
        ----------
        addr : str
            The address string. This has the form 'protocol://interface:port',
            for example 'tcp://127.0.0.1:5555'. Protocols supported are
            tcp, udp, pgm, inproc and ipc. If the address is unicode, it is
            encoded to utf-8 first.
        """
        rc: C.int
        c_addr: p_char
        _check_version((3, 2), 'disconnect')
        _check_closed(self)
        if isinstance(addr, str):
            addr = addr.encode('utf-8')
        if not isinstance(addr, bytes):
            raise TypeError('expected str, got: %r' % addr)
        c_addr = addr
        rc = zmq_disconnect(self.handle, c_addr)
        if rc != 0:
            raise ZMQError()

    def monitor(self, addr, events: C.int=ZMQ_EVENT_ALL):
        """
        Start publishing socket events on inproc.
        See libzmq docs for zmq_monitor for details.

        While this function is available from libzmq 3.2,
        pyzmq cannot parse monitor messages from libzmq prior to 4.0.

        .. versionadded: libzmq-3.2
        .. versionadded: 14.0

        Parameters
        ----------
        addr : str
            The inproc url used for monitoring. Passing None as
            the addr will cause an existing socket monitor to be
            deregistered.
        events : int
            default: zmq.EVENT_ALL
            The zmq event bitmask for which events will be sent to the monitor.
        """
        _check_version((3, 2), 'monitor')
        if isinstance(addr, str):
            addr = addr.encode('utf-8')
        c_addr: p_char
        if addr is None:
            c_addr = NULL
        else:
            try:
                c_addr = addr
            except TypeError:
                raise TypeError(f'Monitor addr must be str, got {addr!r}') from None
        _check_rc(zmq_socket_monitor(self.handle, c_addr, events))

    def join(self, group):
        """
        Join a RADIO-DISH group

        Only for DISH sockets.

        libzmq and pyzmq must have been built with ZMQ_BUILD_DRAFT_API

        .. versionadded:: 17
        """
        _check_version((4, 2), 'RADIO-DISH')
        if not zmq.has('draft'):
            raise RuntimeError('libzmq must be built with draft support')
        if isinstance(group, str):
            group = group.encode('utf8')
        rc: C.int = zmq_join(self.handle, group)
        _check_rc(rc)

    def leave(self, group):
        """
        Leave a RADIO-DISH group

        Only for DISH sockets.

        libzmq and pyzmq must have been built with ZMQ_BUILD_DRAFT_API

        .. versionadded:: 17
        """
        _check_version((4, 2), 'RADIO-DISH')
        if not zmq.has('draft'):
            raise RuntimeError('libzmq must be built with draft support')
        rc: C.int = zmq_leave(self.handle, group)
        _check_rc(rc)

    def send(self, data, flags=0, copy: bint=True, track: bint=False):
        """
        Send a single zmq message frame on this socket.

        This queues the message to be sent by the IO thread at a later time.

        With flags=NOBLOCK, this raises :class:`ZMQError` if the queue is full;
        otherwise, this waits until space is available.
        See :class:`Poller` for more general non-blocking I/O.

        Parameters
        ----------
        data : bytes, Frame, memoryview
            The content of the message. This can be any object that provides
            the Python buffer API (`memoryview(data)` can be called).
        flags : int
            0, NOBLOCK, SNDMORE, or NOBLOCK|SNDMORE.
        copy : bool
            Should the message be sent in a copying or non-copying manner.
        track : bool
            Should the message be tracked for notification that ZMQ has
            finished with it? (ignored if copy=True)

        Returns
        -------
        None : if `copy` or not track
            None if message was sent, raises an exception otherwise.
        MessageTracker : if track and not copy
            a MessageTracker object, whose `done` property will
            be False until the send is completed.

        Raises
        ------
        TypeError
            If a unicode object is passed
        ValueError
            If `track=True`, but an untracked Frame is passed.
        ZMQError
            for any of the reasons zmq_msg_send might fail (including
            if NOBLOCK is set and the outgoing queue is full).

        """
        _check_closed(self)
        if isinstance(data, str):
            raise TypeError('unicode not allowed, use send_string')
        if copy and (not isinstance(data, Frame)):
            return _send_copy(self.handle, data, flags)
        else:
            if isinstance(data, Frame):
                if track and (not data.tracker):
                    raise ValueError('Not a tracked message')
                msg = data
            else:
                if self.copy_threshold:
                    buf = memoryview(data)
                    if buf.nbytes < self.copy_threshold:
                        _send_copy(self.handle, buf, flags)
                        return zmq._FINISHED_TRACKER
                msg = Frame(data, track=track, copy_threshold=self.copy_threshold)
            return _send_frame(self.handle, msg, flags)

    def recv(self, flags=0, copy: bint=True, track: bint=False):
        """
        Receive a message.

        With flags=NOBLOCK, this raises :class:`ZMQError` if no messages have
        arrived; otherwise, this waits until a message arrives.
        See :class:`Poller` for more general non-blocking I/O.

        Parameters
        ----------
        flags : int
            0 or NOBLOCK.
        copy : bool
            Should the message be received in a copying or non-copying manner?
            If False a Frame object is returned, if True a string copy of
            message is returned.
        track : bool
            Should the message be tracked for notification that ZMQ has
            finished with it? (ignored if copy=True)

        Returns
        -------
        msg : bytes or Frame
            The received message frame.  If `copy` is False, then it will be a Frame,
            otherwise it will be bytes.

        Raises
        ------
        ZMQError
            for any of the reasons zmq_msg_recv might fail (including if
            NOBLOCK is set and no new messages have arrived).
        """
        _check_closed(self)
        if copy:
            return _recv_copy(self.handle, flags)
        else:
            frame = _recv_frame(self.handle, flags, track)
            frame.more = self.get(zmq.RCVMORE)
            return frame