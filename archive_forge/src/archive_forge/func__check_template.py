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
def _check_template(self, content):
    dist, cmd = self.get_cmd()
    os.chdir(self.tmp_dir)
    self.write_file('MANIFEST.in', content)
    cmd.ensure_finalized()
    cmd.filelist = FileList()
    cmd.read_template()
    warnings = self.get_logs(WARN)
    self.assertEqual(len(warnings), 1)