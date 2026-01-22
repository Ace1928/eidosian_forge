from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
class TestTcsh(_TestSh, unittest.TestCase):
    expected_failures = ['test_unquoted_space', 'test_quoted_space', 'test_continuation', 'test_parse_special_characters', 'test_parse_special_characters_dollar', 'test_comp_point']

    def setUp(self):
        sh = Shell('tcsh')
        path = ' '.join([os.path.join(BASE_DIR, 'scripts'), TEST_DIR, '$path'])
        sh.run_command('set path = ({0})'.format(path))
        sh.run_command('setenv PYTHONPATH {0}'.format(BASE_DIR))
        output = sh.run_command('eval `register-python-argcomplete --shell tcsh prog`')
        self.assertEqual(output, '')
        self.sh = sh

    def tearDown(self):
        with self.assertRaises((pexpect.EOF, OSError)):
            self.sh.run_command('exit')
            self.sh.run_command('')