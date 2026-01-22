import doctest
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import ClassVar, List
from unittest import SkipTest, expectedFailure, skipIf
from unittest import TestCase as _TestCase
class BlackboxTestCase(TestCase):
    """Blackbox testing."""
    bin_directories: ClassVar[List[str]] = [os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'bin')), '/usr/bin', '/usr/local/bin']

    def bin_path(self, name):
        """Determine the full path of a binary.

        Args:
          name: Name of the script
        Returns: Full path
        """
        for d in self.bin_directories:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
        else:
            raise SkipTest('Unable to find binary %s' % name)

    def run_command(self, name, args):
        """Run a Dulwich command.

        Args:
          name: Name of the command, as it exists in bin/
          args: Arguments to the command
        """
        env = dict(os.environ)
        env['PYTHONPATH'] = os.pathsep.join(sys.path)
        argv = [sys.executable, self.bin_path(name), *args]
        return subprocess.Popen(argv, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE, env=env)