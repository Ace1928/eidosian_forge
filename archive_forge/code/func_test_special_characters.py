from __future__ import absolute_import, division, print_function, unicode_literals
import os, sys, shutil, argparse, subprocess, unittest, io
import pexpect, pexpect.replwrap
from tempfile import TemporaryFile, NamedTemporaryFile, mkdtemp
from argparse import ArgumentParser, SUPPRESS
from argcomplete import (
from argcomplete.completers import FilesCompleter, DirectoriesCompleter, SuppressCompleter
from argcomplete.compat import USING_PYTHON2, str, sys_encoding, ensure_str, ensure_bytes
def test_special_characters(self):
    self.assertEqual(self.sh.run_command('prog spec d\tf'), 'd$e$f\r\n')
    self.assertEqual(self.sh.run_command('prog spec x\t'), 'x!x\r\n')
    self.assertEqual(self.sh.run_command('prog spec y\t'), 'y\\y\r\n')