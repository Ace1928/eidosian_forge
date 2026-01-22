import codecs
import locale
import sys
from typing import Set
from .. import osutils
from . import TestCase
from .ui_testing import BytesIOWithEncoding, StringIOWithEncoding
class TestTerminalEncoding(TestCase):
    """Test the auto-detection of proper terminal encoding."""

    def setUp(self):
        super().setUp()
        self.overrideAttr(sys, 'stdin')
        self.overrideAttr(sys, 'stdout')
        self.overrideAttr(sys, 'stderr')
        self.overrideAttr(osutils, '_cached_user_encoding')

    def make_wrapped_streams(self, stdout_encoding, stderr_encoding, stdin_encoding, user_encoding='user_encoding', enable_fake_encodings=True):
        sys.stdout = StringIOWithEncoding()
        sys.stdout.encoding = stdout_encoding
        sys.stderr = StringIOWithEncoding()
        sys.stderr.encoding = stderr_encoding
        sys.stdin = StringIOWithEncoding()
        sys.stdin.encoding = stdin_encoding
        osutils._cached_user_encoding = user_encoding
        if enable_fake_encodings:
            fake_codec.add(stdout_encoding)
            fake_codec.add(stderr_encoding)
            fake_codec.add(stdin_encoding)

    def test_get_terminal_encoding(self):
        self.make_wrapped_streams('stdout_encoding', 'stderr_encoding', 'stdin_encoding')
        self.assertEqual('stdout_encoding', osutils.get_terminal_encoding())
        sys.stdout.encoding = None
        self.assertEqual('stdin_encoding', osutils.get_terminal_encoding())
        sys.stdin.encoding = None
        self.assertEqual('user_encoding', osutils.get_terminal_encoding())

    def test_get_terminal_encoding_silent(self):
        self.make_wrapped_streams('stdout_encoding', 'stderr_encoding', 'stdin_encoding')
        log = self.get_log()
        osutils.get_terminal_encoding()
        self.assertEqual(log, self.get_log())

    def test_get_terminal_encoding_trace(self):
        self.make_wrapped_streams('stdout_encoding', 'stderr_encoding', 'stdin_encoding')
        log = self.get_log()
        osutils.get_terminal_encoding(trace=True)
        self.assertNotEqual(log, self.get_log())

    def test_terminal_cp0(self):
        self.make_wrapped_streams('cp0', 'cp0', 'cp0', user_encoding='latin-1', enable_fake_encodings=False)
        self.assertEqual('latin-1', osutils.get_terminal_encoding())
        self.assertEqual('', sys.stderr.getvalue())

    def test_terminal_cp_unknown(self):
        self.make_wrapped_streams('cp-unknown', 'cp-unknown', 'cp-unknown', user_encoding='latin-1', enable_fake_encodings=False)
        self.assertEqual('latin-1', osutils.get_terminal_encoding())
        self.assertEqual('brz: warning: unknown terminal encoding cp-unknown.\n  Using encoding latin-1 instead.\n', sys.stderr.getvalue())