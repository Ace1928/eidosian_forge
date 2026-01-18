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
def test_reader_os_root(tmpdir):
    with pytest.raises(ValueError):
        journal.Reader(journal.OS_ROOT)
    with skip_valueerror():
        j1 = journal.Reader(path=tmpdir.strpath, flags=journal.OS_ROOT)
    with skip_valueerror():
        j2 = journal.Reader(path=tmpdir.strpath, flags=journal.OS_ROOT | journal.CURRENT_USER)
    j3 = journal.Reader(path=tmpdir.strpath, flags=journal.OS_ROOT | journal.SYSTEM_ONLY)