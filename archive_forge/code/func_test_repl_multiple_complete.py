from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
def test_repl_multiple_complete(self):
    p = ArgumentParser()
    p.add_argument('--foo')
    p.add_argument('--bar')
    c = CompletionFinder(p, always_complete_options=True)
    completions = self.run_completer(p, c, 'prog ')
    assert set(completions) == set(['-h', '--help', '--foo', '--bar'])
    completions = self.run_completer(p, c, 'prog --')
    assert set(completions) == set(['--help', '--foo', '--bar'])