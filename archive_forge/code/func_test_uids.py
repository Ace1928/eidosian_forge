from __future__ import print_function
import select
import contextlib
import errno
from systemd import login
import pytest
def test_uids():
    with skip_oserror(errno.ENOENT):
        uids = login.uids()
        assert len(uids) >= 0