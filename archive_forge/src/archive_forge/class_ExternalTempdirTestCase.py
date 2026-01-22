import glob
import operator
import os
import shutil
import sys
import tempfile
from incremental import Version
from twisted.python import release
from twisted.python._release import (
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
class ExternalTempdirTestCase(TestCase):
    """
    A test case which has mkdir make directories outside of the usual spot, so
    that Git commands don't interfere with the Twisted checkout.
    """

    def mktemp(self):
        """
        Make our own directory.
        """
        newDir = tempfile.mkdtemp(dir=tempfile.gettempdir())
        self.addCleanup(shutil.rmtree, newDir)
        return newDir