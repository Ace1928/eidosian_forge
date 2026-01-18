import contextlib
import io
import os
import sys
import shutil
import subprocess
import tempfile
from pyflakes.checker import PYPY
from pyflakes.messages import UnusedImport
from pyflakes.reporter import Reporter
from pyflakes.api import (
from pyflakes.test.harness import TestCase, skipIf
def test_readFromStdin(self):
    """
        If no arguments are passed to C{pyflakes} then it reads from stdin.
        """
    d = self.runPyflakes([], stdin='import contraband')
    expected = UnusedImport('<stdin>', Node(1), 'contraband')
    self.assertEqual(d, (f'{expected}{os.linesep}', '', 1))