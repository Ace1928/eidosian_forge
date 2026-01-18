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
def test_seek_realtime(tmpdir):
    j = journal.Reader(path=tmpdir.strpath)
    now = time.time()
    j.seek_realtime(now)
    j.seek_realtime(12345)
    long_ago = datetime.datetime(1970, 5, 4)
    j.seek_realtime(long_ago)