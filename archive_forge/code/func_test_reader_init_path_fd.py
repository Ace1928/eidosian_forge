from __future__ import print_function
import contextlib
import datetime
import errno
import logging
import os
import time
import uuid
import sys
import traceback
from systemd import journal, id128
from systemd.journal import _make_line
import pytest
def test_reader_init_path_fd(tmpdir):
    fd = os.open(tmpdir.strpath, os.O_RDONLY)
    with skip_oserror(errno.ENOSYS):
        j1 = journal.Reader(path=fd)
    assert list(j1) == []
    with skip_valueerror():
        j2 = journal.Reader(journal.SYSTEM, path=fd)
    assert list(j2) == []
    j3 = journal.Reader(journal.CURRENT_USER, path=fd)
    assert list(j3) == []