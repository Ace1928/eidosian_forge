import fcntl
import os
import pty
import struct
import sys
import termios
import textwrap
import unittest
from bpython.test import TEST_CONFIG
from bpython.config import getpreferredencoding
@unittest.skipUnless(have_urwid, 'urwid is required')
@unittest.skipIf(reactor is None, 'twisted is not available')
class UrwidCrashersTest(TrialTestCase, CrashersTest):
    backend = 'urwid'