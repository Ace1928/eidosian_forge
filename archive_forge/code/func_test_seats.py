from __future__ import print_function
import select
import contextlib
import errno
from systemd import login
import pytest
def test_seats():
    with skip_oserror(errno.ENOENT):
        seats = login.seats()
        assert len(seats) >= 0