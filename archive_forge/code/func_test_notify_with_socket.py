import sys
import os
import posix
import socket
import contextlib
import errno
from systemd.daemon import (booted,
import pytest
def test_notify_with_socket(tmpdir):
    path = tmpdir.join('socket').strpath
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        sock.bind(path)
    except socket.error as e:
        pytest.xfail('failed to bind socket (%s)' % e)
    SO_PASSCRED = getattr(socket, 'SO_PASSCRED', 16)
    sock.setsockopt(socket.SOL_SOCKET, SO_PASSCRED, 1)
    os.environ['NOTIFY_SOCKET'] = path
    assert notify('READY=1')
    with skip_enosys():
        assert notify('FDSTORE=1', fds=[])
    assert notify('FDSTORE=1', fds=[1, 2])
    assert notify('FDSTORE=1', pid=os.getpid())
    assert notify('FDSTORE=1', pid=os.getpid(), fds=(1,))