from ._cares import ffi as _ffi, lib as _lib
import _cffi_backend  # hint for bundler tools
from . import errno
from .utils import ascii_bytes, maybe_str, parse_name
from ._version import __version__
import collections.abc
import socket
import math
import functools
import sys
class Channel:
    __qtypes__ = (_lib.T_A, _lib.T_AAAA, _lib.T_ANY, _lib.T_CAA, _lib.T_CNAME, _lib.T_MX, _lib.T_NAPTR, _lib.T_NS, _lib.T_PTR, _lib.T_SOA, _lib.T_SRV, _lib.T_TXT)
    __qclasses__ = (_lib.C_IN, _lib.C_CHAOS, _lib.C_HS, _lib.C_NONE, _lib.C_ANY)

    def __init__(self, flags=None, timeout=None, tries=None, ndots=None, tcp_port=None, udp_port=None, servers=None, domains=None, lookups=None, sock_state_cb=None, socket_send_buffer_size=None, socket_receive_buffer_size=None, rotate=False, local_ip=None, local_dev=None, resolvconf_path=None):
        channel = _ffi.new('ares_channel *')
        options = _ffi.new('struct ares_options *')
        optmask = 0
        if flags is not None:
            options.flags = flags
            optmask = optmask | _lib.ARES_OPT_FLAGS
        if timeout is not None:
            options.timeout = int(timeout * 1000)
            optmask = optmask | _lib.ARES_OPT_TIMEOUTMS
        if tries is not None:
            options.tries = tries
            optmask = optmask | _lib.ARES_OPT_TRIES
        if ndots is not None:
            options.ndots = ndots
            optmask = optmask | _lib.ARES_OPT_NDOTS
        if tcp_port is not None:
            options.tcp_port = tcp_port
            optmask = optmask | _lib.ARES_OPT_TCP_PORT
        if udp_port is not None:
            options.udp_port = udp_port
            optmask = optmask | _lib.ARES_OPT_UDP_PORT
        if socket_send_buffer_size is not None:
            options.socket_send_buffer_size = socket_send_buffer_size
            optmask = optmask | _lib.ARES_OPT_SOCK_SNDBUF
        if socket_receive_buffer_size is not None:
            options.socket_receive_buffer_size = socket_receive_buffer_size
            optmask = optmask | _lib.ARES_OPT_SOCK_RCVBUF
        if sock_state_cb:
            if not callable(sock_state_cb):
                raise TypeError('sock_state_cb is not callable')
            userdata = _ffi.new_handle(sock_state_cb)
            self._sock_state_cb_handle = userdata
            options.sock_state_cb = _lib._sock_state_cb
            options.sock_state_cb_data = userdata
            optmask = optmask | _lib.ARES_OPT_SOCK_STATE_CB
        if lookups:
            options.lookups = _ffi.new('char[]', ascii_bytes(lookups))
            optmask = optmask | _lib.ARES_OPT_LOOKUPS
        if domains:
            strs = [_ffi.new('char[]', ascii_bytes(i)) for i in domains]
            c = _ffi.new('char *[%d]' % (len(domains) + 1))
            for i in range(len(domains)):
                c[i] = strs[i]
            options.domains = c
            options.ndomains = len(domains)
            optmask = optmask | _lib.ARES_OPT_DOMAINS
        if rotate:
            optmask = optmask | _lib.ARES_OPT_ROTATE
        if resolvconf_path is not None:
            optmask = optmask | _lib.ARES_OPT_RESOLVCONF
            options.resolvconf_path = _ffi.new('char[]', ascii_bytes(resolvconf_path))
        r = _lib.ares_init_options(channel, options, optmask)
        if r != _lib.ARES_SUCCESS:
            raise AresError('Failed to initialize c-ares channel')
        self._channel = _ffi.gc(channel, lambda x: _lib.ares_destroy(x[0]))
        if servers:
            self.servers = servers
        if local_ip:
            self.set_local_ip(local_ip)
        if local_dev:
            self.set_local_dev(local_dev)

    def cancel(self):
        _lib.ares_cancel(self._channel[0])

    @property
    def servers(self):
        servers = _ffi.new('struct ares_addr_node **')
        r = _lib.ares_get_servers(self._channel[0], servers)
        if r != _lib.ARES_SUCCESS:
            raise AresError(r, errno.strerror(r))
        server_list = []
        server = _ffi.new('struct ares_addr_node **', servers[0])
        while True:
            if server == _ffi.NULL:
                break
            ip = _ffi.new('char []', _lib.INET6_ADDRSTRLEN)
            s = server[0]
            if _ffi.NULL != _lib.ares_inet_ntop(s.family, _ffi.addressof(s.addr), ip, _lib.INET6_ADDRSTRLEN):
                server_list.append(maybe_str(_ffi.string(ip, _lib.INET6_ADDRSTRLEN)))
            server = s.next
        return server_list

    @servers.setter
    def servers(self, servers):
        c = _ffi.new('struct ares_addr_node[%d]' % len(servers))
        for i, server in enumerate(servers):
            if _lib.ares_inet_pton(socket.AF_INET, ascii_bytes(server), _ffi.addressof(c[i].addr.addr4)) == 1:
                c[i].family = socket.AF_INET
            elif _lib.ares_inet_pton(socket.AF_INET6, ascii_bytes(server), _ffi.addressof(c[i].addr.addr6)) == 1:
                c[i].family = socket.AF_INET6
            else:
                raise ValueError('invalid IP address')
            if i > 0:
                c[i - 1].next = _ffi.addressof(c[i])
        r = _lib.ares_set_servers(self._channel[0], c)
        if r != _lib.ARES_SUCCESS:
            raise AresError(r, errno.strerror(r))

    def getsock(self):
        rfds = []
        wfds = []
        socks = _ffi.new('ares_socket_t [%d]' % _lib.ARES_GETSOCK_MAXNUM)
        bitmask = _lib.ares_getsock(self._channel[0], socks, _lib.ARES_GETSOCK_MAXNUM)
        for i in range(_lib.ARES_GETSOCK_MAXNUM):
            if _lib.ARES_GETSOCK_READABLE(bitmask, i):
                rfds.append(socks[i])
            if _lib.ARES_GETSOCK_WRITABLE(bitmask, i):
                wfds.append(socks[i])
        return (rfds, wfds)

    def process_fd(self, read_fd, write_fd):
        _lib.ares_process_fd(self._channel[0], _ffi.cast('ares_socket_t', read_fd), _ffi.cast('ares_socket_t', write_fd))

    def timeout(self, t=None):
        maxtv = _ffi.NULL
        tv = _ffi.new('struct timeval*')
        if t is not None:
            if t >= 0.0:
                maxtv = _ffi.new('struct timeval*')
                maxtv.tv_sec = int(math.floor(t))
                maxtv.tv_usec = int(math.fmod(t, 1.0) * 1000000)
            else:
                raise ValueError('timeout needs to be a positive number or None')
        _lib.ares_timeout(self._channel[0], maxtv, tv)
        if tv == _ffi.NULL:
            return 0.0
        return tv.tv_sec + tv.tv_usec / 1000000.0

    def gethostbyaddr(self, addr, callback):
        if not callable(callback):
            raise TypeError('a callable is required')
        addr4 = _ffi.new('struct in_addr*')
        addr6 = _ffi.new('struct ares_in6_addr*')
        if _lib.ares_inet_pton(socket.AF_INET, ascii_bytes(addr), addr4) == 1:
            address = addr4
            family = socket.AF_INET
        elif _lib.ares_inet_pton(socket.AF_INET6, ascii_bytes(addr), addr6) == 1:
            address = addr6
            family = socket.AF_INET6
        else:
            raise ValueError('invalid IP address')
        userdata = _ffi.new_handle(callback)
        _global_set.add(userdata)
        _lib.ares_gethostbyaddr(self._channel[0], address, _ffi.sizeof(address[0]), family, _lib._host_cb, userdata)

    def gethostbyname(self, name, family, callback):
        if not callable(callback):
            raise TypeError('a callable is required')
        userdata = _ffi.new_handle(callback)
        _global_set.add(userdata)
        _lib.ares_gethostbyname(self._channel[0], parse_name(name), family, _lib._host_cb, userdata)

    def getaddrinfo(self, host, port, callback, family=0, type=0, proto=0, flags=0):
        if not callable(callback):
            raise TypeError('a callable is required')
        if port is None:
            service = _ffi.NULL
        elif isinstance(port, int):
            service = str(port).encode('ascii')
        else:
            service = ascii_bytes(port)
        userdata = _ffi.new_handle(callback)
        _global_set.add(userdata)
        hints = _ffi.new('struct ares_addrinfo_hints*')
        hints.ai_flags = flags
        hints.ai_family = family
        hints.ai_socktype = type
        hints.ai_protocol = proto
        _lib.ares_getaddrinfo(self._channel[0], parse_name(host), service, hints, _lib._addrinfo_cb, userdata)

    def query(self, name, query_type, callback, query_class=None):
        self._do_query(_lib.ares_query, name, query_type, callback, query_class=query_class)

    def search(self, name, query_type, callback, query_class=None):
        self._do_query(_lib.ares_search, name, query_type, callback, query_class=query_class)

    def _do_query(self, func, name, query_type, callback, query_class=None):
        if not callable(callback):
            raise TypeError('a callable is required')
        if query_type not in self.__qtypes__:
            raise ValueError('invalid query type specified')
        if query_class is None:
            query_class = _lib.C_IN
        if query_class not in self.__qclasses__:
            raise ValueError('invalid query class specified')
        userdata = _ffi.new_handle((callback, query_type))
        _global_set.add(userdata)
        func(self._channel[0], parse_name(name), query_class, query_type, _lib._query_cb, userdata)

    def set_local_ip(self, ip):
        addr4 = _ffi.new('struct in_addr*')
        addr6 = _ffi.new('struct ares_in6_addr*')
        if _lib.ares_inet_pton(socket.AF_INET, ascii_bytes(ip), addr4) == 1:
            _lib.ares_set_local_ip4(self._channel[0], socket.ntohl(addr4.s_addr))
        elif _lib.ares_inet_pton(socket.AF_INET6, ascii_bytes(ip), addr6) == 1:
            _lib.ares_set_local_ip6(self._channel[0], addr6)
        else:
            raise ValueError('invalid IP address')

    def getnameinfo(self, address, flags, callback):
        if not callable(callback):
            raise TypeError('a callable is required')
        if len(address) == 2:
            ip, port = address
            sa4 = _ffi.new('struct sockaddr_in*')
            if _lib.ares_inet_pton(socket.AF_INET, ascii_bytes(ip), _ffi.addressof(sa4.sin_addr)) != 1:
                raise ValueError('Invalid IPv4 address %r' % ip)
            sa4.sin_family = socket.AF_INET
            sa4.sin_port = socket.htons(port)
            sa = sa4
        elif len(address) == 4:
            ip, port, flowinfo, scope_id = address
            sa6 = _ffi.new('struct sockaddr_in6*')
            if _lib.ares_inet_pton(socket.AF_INET6, ascii_bytes(ip), _ffi.addressof(sa6.sin6_addr)) != 1:
                raise ValueError('Invalid IPv6 address %r' % ip)
            sa6.sin6_family = socket.AF_INET6
            sa6.sin6_port = socket.htons(port)
            sa6.sin6_flowinfo = socket.htonl(flowinfo)
            sa6.sin6_scope_id = scope_id
            sa = sa6
        else:
            raise ValueError('Invalid address argument')
        userdata = _ffi.new_handle(callback)
        _global_set.add(userdata)
        _lib.ares_getnameinfo(self._channel[0], _ffi.cast('struct sockaddr*', sa), _ffi.sizeof(sa[0]), flags, _lib._nameinfo_cb, userdata)

    def set_local_dev(self, dev):
        _lib.ares_set_local_dev(self._channel[0], dev)