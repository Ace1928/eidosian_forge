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
def test_reader_this_machine(tmpdir):
    j = journal.Reader(path=tmpdir.strpath)
    with j:
        j.this_machine()
        j.this_machine(TEST_MID)
        j.this_machine(TEST_MID.hex)