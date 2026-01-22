from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
class TestBash(_TestSh, unittest.TestCase):
    expected_failures = ['test_parse_special_characters_dollar', 'test_exclamation_in_double_quotes']
    if BASH_MAJOR_VERSION < 4:
        expected_failures.append('test_quoted_exact')
    install_cmd = 'eval "$(register-python-argcomplete prog)"'

    def setUp(self):
        sh = pexpect.replwrap.bash()
        path = ':'.join([os.path.join(BASE_DIR, 'scripts'), TEST_DIR, '$PATH'])
        sh.run_command('export PATH={0}'.format(path))
        sh.run_command('export PYTHONPATH={0}'.format(BASE_DIR))
        sh.run_command('complete -r python python2 python3')
        output = sh.run_command(self.install_cmd)
        self.assertEqual(output, '')
        self.sh = sh

    def test_one_space_after_exact(self):
        """Test exactly one space is appended after an exact match."""
        result = self.sh.run_command('prog basic f\t"\x01echo "')
        self.assertEqual(result, 'prog basic foo \r\n')

    def test_debug_output(self):
        self.assertEqual(self.sh.run_command('prog debug f\t'), 'foo\r\n')
        self.sh.run_command('export _ARC_DEBUG=1')
        output = self.sh.run_command('prog debug f\t')
        self.assertIn('PYTHON_ARGCOMPLETE_STDOUT\r\n', output)
        self.assertIn('PYTHON_ARGCOMPLETE_STDERR\r\n', output)
        self.assertTrue(output.endswith('foo\r\n'))