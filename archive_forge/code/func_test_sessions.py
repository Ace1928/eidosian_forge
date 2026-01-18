from __future__ import print_function
import select
import contextlib
import errno
from systemd import login
import pytest
def test_sessions():
    with skip_oserror(errno.ENOENT):
        sessions = login.sessions()
        assert len(sessions) >= 0