import email
import email.errors
import os
import re
import sysconfig
import tempfile
import textwrap
import fixtures
import pkg_resources
import six
import testscenarios
import testtools
from testtools import matchers
import virtualenv
from wheel import wheelfile
from pbr import git
from pbr import packaging
from pbr.tests import base
class Venv(fixtures.Fixture):
    """Create a virtual environment for testing with.

    :attr path: The path to the environment root.
    :attr python: The path to the python binary in the environment.
    """

    def __init__(self, reason, modules=(), pip_cmd=None):
        """Create a Venv fixture.

        :param reason: A human readable string to bake into the venv
            file path to aid diagnostics in the case of failures.
        :param modules: A list of modules to install, defaults to latest
            pip, wheel, and the working copy of PBR.
        :attr pip_cmd: A list to override the default pip_cmd passed to
            python for installing base packages.
        """
        self._reason = reason
        if modules == ():
            modules = ['pip', 'wheel', 'build', PBR_ROOT]
        self.modules = modules
        if pip_cmd is None:
            self.pip_cmd = ['-m', 'pip', '-v', 'install']
        else:
            self.pip_cmd = pip_cmd

    def _setUp(self):
        path = self.useFixture(fixtures.TempDir()).path
        virtualenv.cli_run([path])
        python = os.path.join(path, 'bin', 'python')
        command = [python] + self.pip_cmd + ['-U']
        if self.modules and len(self.modules) > 0:
            command.extend(self.modules)
            self.useFixture(base.CapturedSubprocess('mkvenv-' + self._reason, command))
        self.addCleanup(delattr, self, 'path')
        self.addCleanup(delattr, self, 'python')
        self.path = path
        self.python = python
        return (path, python)