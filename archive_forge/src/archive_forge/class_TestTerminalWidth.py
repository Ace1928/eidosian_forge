import errno
import os
import select
import socket
import sys
import tempfile
import time
from io import BytesIO
from .. import errors, osutils, tests, trace, win32utils
from . import features, file_utils, test__walkdirs_win32
from .scenarios import load_tests_apply_scenarios
class TestTerminalWidth(tests.TestCase):

    def setUp(self):
        super().setUp()
        self._orig_terminal_size_state = osutils._terminal_size_state
        self._orig_first_terminal_size = osutils._first_terminal_size
        self.addCleanup(self.restore_osutils_globals)
        osutils._terminal_size_state = 'no_data'
        osutils._first_terminal_size = None

    def restore_osutils_globals(self):
        osutils._terminal_size_state = self._orig_terminal_size_state
        osutils._first_terminal_size = self._orig_first_terminal_size

    def replace_stdout(self, new):
        self.overrideAttr(sys, 'stdout', new)

    def replace__terminal_size(self, new):
        self.overrideAttr(osutils, '_terminal_size', new)

    def set_fake_tty(self):

        class I_am_a_tty:

            def isatty(self):
                return True
        self.replace_stdout(I_am_a_tty())

    def test_default_values(self):
        self.assertEqual(80, osutils.default_terminal_width)

    def test_defaults_to_BRZ_COLUMNS(self):
        self.assertNotEqual('12', os.environ['BRZ_COLUMNS'])
        self.overrideEnv('BRZ_COLUMNS', '12')
        self.assertEqual(12, osutils.terminal_width())

    def test_BRZ_COLUMNS_0_no_limit(self):
        self.overrideEnv('BRZ_COLUMNS', '0')
        self.assertEqual(None, osutils.terminal_width())

    def test_falls_back_to_COLUMNS(self):
        self.overrideEnv('BRZ_COLUMNS', None)
        self.assertNotEqual('42', os.environ['COLUMNS'])
        self.set_fake_tty()
        self.overrideEnv('COLUMNS', '42')
        self.assertEqual(42, osutils.terminal_width())

    def test_tty_default_without_columns(self):
        self.overrideEnv('BRZ_COLUMNS', None)
        self.overrideEnv('COLUMNS', None)

        def terminal_size(w, h):
            return (42, 42)
        self.set_fake_tty()
        self.replace__terminal_size(terminal_size)
        self.assertEqual(42, osutils.terminal_width())

    def test_non_tty_default_without_columns(self):
        self.overrideEnv('BRZ_COLUMNS', None)
        self.overrideEnv('COLUMNS', None)
        self.replace_stdout(None)
        self.assertEqual(None, osutils.terminal_width())

    def test_no_TIOCGWINSZ(self):
        self.requireFeature(term_ios_feature)
        termios = term_ios_feature.module
        try:
            termios.TIOCGWINSZ
        except AttributeError:
            pass
        else:
            self.overrideAttr(termios, 'TIOCGWINSZ')
            del termios.TIOCGWINSZ
        self.overrideEnv('BRZ_COLUMNS', None)
        self.overrideEnv('COLUMNS', None)
        osutils.terminal_width()