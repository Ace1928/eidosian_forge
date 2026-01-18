from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
def test_positional_remainder(self):

    def make_parser():
        parser = ArgumentParser()
        parser.add_argument('--foo', choices=['foo1', 'foo2'])
        parser.add_argument('remainder', choices=['pos', '--opt'], nargs=argparse.REMAINDER)
        return parser
    options = ['--foo', '-h', '--help']
    expected_outputs = (('prog ', ['pos', '--opt'] + options), ('prog --foo foo1 ', ['pos', '--opt'] + options), ('prog pos ', ['pos', '--opt']), ('prog -- ', ['pos', '--opt']), ('prog -- --opt ', ['pos', '--opt']))
    for cmd, output in expected_outputs:
        self.assertEqual(set(self.run_completer(make_parser(), cmd)), set(output))