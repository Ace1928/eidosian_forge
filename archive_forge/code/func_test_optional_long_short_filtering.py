from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
def test_optional_long_short_filtering(self):

    def make_parser():
        parser = ArgumentParser()
        parser.add_argument('--foo')
        parser.add_argument('-b', '--bar')
        parser.add_argument('--baz', '--xyz')
        parser.add_argument('-t')
        parser.add_argument('-z', '--zzz')
        parser.add_argument('-x')
        return parser
    long_opts = '--foo --bar --baz --xyz --zzz --help -x -t'.split()
    short_opts = '-b -t -x -z -h --foo --baz --xyz'.split()
    expected_outputs = (('prog ', {'long': long_opts, 'short': short_opts, True: long_opts + short_opts, False: ['']}), ('prog --foo', {'long': ['--foo '], 'short': ['--foo '], True: ['--foo '], False: ['--foo ']}), ('prog --b', {'long': ['--bar', '--baz'], 'short': ['--bar', '--baz'], True: ['--bar', '--baz'], False: ['--bar', '--baz']}), ('prog -z -x', {'long': ['-x '], 'short': ['-x '], True: ['-x '], False: ['-x ']}))
    for cmd, outputs in expected_outputs:
        for always_complete_options, output in outputs.items():
            result = self.run_completer(make_parser(), cmd, always_complete_options=always_complete_options)
            self.assertEqual(set(result), set(output))