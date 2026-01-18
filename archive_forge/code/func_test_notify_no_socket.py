import sys
import os
import posix
import socket
import contextlib
import errno
from systemd.daemon import (booted,
import pytest
def test_notify_no_socket():
    os.environ.pop('NOTIFY_SOCKET', None)
    assert notify('READY=1') is False
    with skip_enosys():
        assert notify('FDSTORE=1', fds=[]) is False
    assert notify('FDSTORE=1', fds=[1, 2]) is False
    assert notify('FDSTORE=1', pid=os.getpid()) is False
    assert notify('FDSTORE=1', pid=os.getpid(), fds=(1,)) is False