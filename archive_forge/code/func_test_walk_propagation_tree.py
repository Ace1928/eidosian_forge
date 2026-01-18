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
def test_walk_propagation_tree(self):
    """Make sure walk_propagation_tree() properly walks the tree of loggers."""
    root, parent, child, grand_child = self.get_logger_tree()
    loggers = list(walk_propagation_tree(grand_child))
    assert loggers == [grand_child, child, parent, root]
    child.propagate = False
    loggers = list(walk_propagation_tree(grand_child))
    assert loggers == [grand_child, child]