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
def test_reader_messageid_match(tmpdir):
    j = journal.Reader(path=tmpdir.strpath)
    with j:
        j.messageid_match(id128.SD_MESSAGE_JOURNAL_START)
        j.messageid_match(id128.SD_MESSAGE_JOURNAL_STOP.hex)