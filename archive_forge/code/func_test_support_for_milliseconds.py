import contextlib
import logging
import logging.handlers
import os
import re
import subprocess
import sys
import tempfile
from humanfriendly.compat import StringIO
from humanfriendly.terminal import ANSI_COLOR_CODES, ANSI_CSI, ansi_style, ansi_wrap
from humanfriendly.testing import PatchedAttribute, PatchedItem, TestCase, retry
from humanfriendly.text import format, random_string
import coloredlogs
import coloredlogs.cli
from coloredlogs import (
from coloredlogs.demo import demonstrate_colored_logging
from coloredlogs.syslog import SystemLogging, is_syslog_supported, match_syslog_handler
from coloredlogs.converter import (
from capturer import CaptureOutput
from verboselogs import VerboseLogger
def test_support_for_milliseconds(self):
    """Make sure milliseconds are hidden by default but can be easily enabled."""
    stream = StringIO()
    install(reconfigure=True, stream=stream)
    logging.info('This should not include milliseconds.')
    assert all(map(PLAIN_TEXT_PATTERN.match, stream.getvalue().splitlines()))
    stream = StringIO()
    install(milliseconds=True, reconfigure=True, stream=stream)
    logging.info('This should include milliseconds.')
    assert all(map(PATTERN_INCLUDING_MILLISECONDS.match, stream.getvalue().splitlines()))