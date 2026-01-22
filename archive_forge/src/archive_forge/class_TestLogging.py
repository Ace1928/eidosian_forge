import errno
import logging
import os
import re
import sys
import tempfile
from io import StringIO
from .. import debug, errors, trace
from ..trace import (_rollover_trace_maybe, be_quiet, get_verbosity_level,
from . import TestCase, TestCaseInTempDir, TestSkipped, features
class TestLogging(TestCase):
    """Check logging functionality robustly records information"""

    def test_note(self):
        trace.note('Noted')
        self.assertEqual('    INFO  Noted\n', self.get_log())

    def test_warning(self):
        trace.warning('Warned')
        self.assertEqual(' WARNING  Warned\n', self.get_log())

    def test_log(self):
        logging.getLogger('brz').error('Errored')
        self.assertEqual('   ERROR  Errored\n', self.get_log())

    def test_log_sub(self):
        logging.getLogger('brz.test_log_sub').debug('Whispered')
        self.assertEqual('   DEBUG  Whispered\n', self.get_log())

    def test_log_unicode_msg(self):
        logging.getLogger('brz').debug('§')
        self.assertEqual('   DEBUG  §\n', self.get_log())

    def test_log_unicode_arg(self):
        logging.getLogger('brz').debug('%s', '§')
        self.assertEqual('   DEBUG  §\n', self.get_log())

    def test_log_utf8_msg(self):
        logging.getLogger('brz').debug(b'\xc2\xa7')
        self.assertEqual('   DEBUG  §\n', self.get_log())

    def test_log_utf8_arg(self):
        logging.getLogger('brz').debug(b'%s', b'\xc2\xa7')
        expected = "   DEBUG  b'\\xc2\\xa7'\n"
        self.assertEqual(expected, self.get_log())

    def test_log_bytes_msg(self):
        logging.getLogger('brz').debug(b'\xa7')
        log = self.get_log()
        self.assertContainsString(log, 'UnicodeDecodeError: ')
        self.assertContainsRe(log, "Logging record unformattable: b?'\\\\xa7' % \\(\\)\n")

    def test_log_bytes_arg(self):
        logging.getLogger('brz').debug(b'%s', b'\xa7')
        log = self.get_log()
        self.assertEqual("   DEBUG  b'\\xa7'\n", self.get_log())

    def test_log_mixed_strings(self):
        logging.getLogger('brz').debug('%s', b'\xa7')
        log = self.get_log()
        self.assertEqual("   DEBUG  b'\\xa7'\n", self.get_log())

    def test_log_repr_broken(self):

        class BadRepr:

            def __repr__(self):
                raise ValueError('Broken object')
        logging.getLogger('brz').debug('%s', BadRepr())
        log = self.get_log()
        self.assertContainsRe(log, 'ValueError: Broken object\n')
        self.assertContainsRe(log, "Logging record unformattable: '%s' % .*\n")