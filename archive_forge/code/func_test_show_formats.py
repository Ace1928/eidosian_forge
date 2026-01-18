import os
import tarfile
import unittest
import warnings
import zipfile
from os.path import join
from textwrap import dedent
from test.support import captured_stdout
from test.support.warnings_helper import check_warnings
from distutils.command.sdist import sdist, show_formats
from distutils.core import Distribution
from distutils.tests.test_config import BasePyPIRCCommandTestCase
from distutils.errors import DistutilsOptionError
from distutils.spawn import find_executable
from distutils.log import WARN
from distutils.filelist import FileList
from distutils.archive_util import ARCHIVE_FORMATS
from distutils.core import setup
import somecode
def test_show_formats(self):
    with captured_stdout() as stdout:
        show_formats()
    num_formats = len(ARCHIVE_FORMATS.keys())
    output = [line for line in stdout.getvalue().split('\n') if line.strip().startswith('--formats=')]
    self.assertEqual(len(output), num_formats)