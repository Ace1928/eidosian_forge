import sys ; sys.path.insert(0, '..')
import DNS
import socket
import unittest
def addAAAA(p, name, klass, ttl, address):
    """Add AAAA record to a packer.
    """
    addr_buf = socket.inet_pton(socket.AF_INET6, address)
    p.addRRheader(name, DNS.Type.AAAA, klass, ttl)
    p.buf = p.buf + addr_buf
    p.endRR()
    return p