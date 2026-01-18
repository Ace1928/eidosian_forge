import io
import distutils.core
import os
import shutil
import sys
from test.support import captured_stdout
from test.support import os_helper
import unittest
from distutils.tests import support
from distutils import log
from distutils.core import setup
import os
from distutils.core import setup
from distutils.core import setup
from distutils.core import setup
from distutils.command.install import install as _install
def test_run_setup_uses_current_dir(self):
    sys.stdout = io.StringIO()
    cwd = os.getcwd()
    os.mkdir(os_helper.TESTFN)
    setup_py = os.path.join(os_helper.TESTFN, 'setup.py')
    distutils.core.run_setup(self.write_setup(setup_prints_cwd, path=setup_py))
    output = sys.stdout.getvalue()
    if output.endswith('\n'):
        output = output[:-1]
    self.assertEqual(cwd, output)