import platform
import socket
import struct
from os_ken.lib import sockaddr
def _set_tcp_md5sig_linux(s, addr, key):
    af = s.family
    if af == socket.AF_INET:
        sa = sockaddr.sa_in4(addr)
    elif af == socket.AF_INET6:
        sa = sockaddr.sa_in6(addr)
    else:
        raise ValueError('unsupported af %s' % (af,))
    ss = sockaddr.sa_to_ss(sa)
    tcp_md5sig = ss + struct.pack('2xH4x80s', len(key), key)
    s.setsockopt(socket.IPPROTO_TCP, TCP_MD5SIG_LINUX, tcp_md5sig)