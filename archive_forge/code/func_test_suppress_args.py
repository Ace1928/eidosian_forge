from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
def test_suppress_args(self):

    def make_parser():
        parser = ArgumentParser()
        parser.add_argument('--foo')
        parser.add_argument('--bar', help=SUPPRESS)
        return parser
    expected_outputs = (('prog ', ['--foo', '-h', '--help']), ('prog --b', ['']))
    for cmd, output in expected_outputs:
        self.assertEqual(set(self.run_completer(make_parser(), cmd)), set(output))
    expected_outputs = (('prog ', ['--foo', '--bar', '-h', '--help']), ('prog --b', ['--bar ']))
    for cmd, output in expected_outputs:
        self.assertEqual(set(self.run_completer(make_parser(), cmd, print_suppressed=True)), set(output))