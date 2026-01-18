import sys
import os
import posix
import socket
import contextlib
import errno
from systemd.daemon import (booted,
import pytest
def test_notify_bad_socket():
    os.environ['NOTIFY_SOCKET'] = '/dev/null'
    with pytest.raises(connection_error):
        notify('READY=1')
    with pytest.raises(connection_error):
        with skip_enosys():
            notify('FDSTORE=1', fds=[])
    with pytest.raises(connection_error):
        notify('FDSTORE=1', fds=[1, 2])
    with pytest.raises(connection_error):
        notify('FDSTORE=1', pid=os.getpid())
    with pytest.raises(connection_error):
        notify('FDSTORE=1', pid=os.getpid(), fds=(1,))