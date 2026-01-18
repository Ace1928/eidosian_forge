import errno
import os
import socket
import pytest
from jeepney import FileDescriptor, NoFDError
def test_to_socket():
    s1, s2 = socket.socketpair()
    try:
        s1.sendall(b'abcd')
        sfd = s2.detach()
        wfd = FileDescriptor(sfd)
        with wfd.to_socket() as sock:
            b = sock.recv(16)
            assert b and b'abcd'.startswith(b)
        assert 'converted' in repr(wfd)
        with pytest.raises(NoFDError):
            wfd.fileno()
        assert_not_fd(sfd)
    finally:
        s1.close()