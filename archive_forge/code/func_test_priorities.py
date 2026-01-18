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
def test_priorities():
    p = journal.JournalHandler.map_priority
    assert p(logging.NOTSET) == journal.LOG_DEBUG
    assert p(logging.DEBUG) == journal.LOG_DEBUG
    assert p(logging.DEBUG - 1) == journal.LOG_DEBUG
    assert p(logging.DEBUG + 1) == journal.LOG_INFO
    assert p(logging.INFO - 1) == journal.LOG_INFO
    assert p(logging.INFO) == journal.LOG_INFO
    assert p(logging.INFO + 1) == journal.LOG_WARNING
    assert p(logging.WARN - 1) == journal.LOG_WARNING
    assert p(logging.WARN) == journal.LOG_WARNING
    assert p(logging.WARN + 1) == journal.LOG_ERR
    assert p(logging.ERROR - 1) == journal.LOG_ERR
    assert p(logging.ERROR) == journal.LOG_ERR
    assert p(logging.ERROR + 1) == journal.LOG_CRIT
    assert p(logging.FATAL) == journal.LOG_CRIT
    assert p(logging.CRITICAL) == journal.LOG_CRIT
    assert p(logging.CRITICAL + 1) == journal.LOG_ALERT