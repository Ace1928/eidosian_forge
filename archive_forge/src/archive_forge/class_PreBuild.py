from __future__ import (absolute_import, division, print_function)
import resource
import base64
import contextlib
import errno
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
class PreBuild:
    """Parsed pre-build instructions."""

    def __init__(self, requirement):
        self.requirement = requirement
        self.constraints = []

    def execute(self, pip):
        """Execute these pre-build instructions."""
        tempdir = tempfile.mkdtemp(prefix='ansible-test-', suffix='-pre-build')
        try:
            options = common_pip_options()
            options.append(self.requirement)
            constraints = '\n'.join(self.constraints) + '\n'
            constraints_path = os.path.join(tempdir, 'constraints.txt')
            write_text_file(constraints_path, constraints, True)
            env = common_pip_environment()
            env.update(PIP_CONSTRAINT=constraints_path)
            command = [sys.executable, pip, 'wheel'] + options
            execute_command(command, env=env, cwd=tempdir)
        finally:
            remove_tree(tempdir)