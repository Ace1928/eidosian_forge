from __future__ import print_function
import os
import fixtures
from pbr import git
from pbr import options
from pbr.tests import base
def _fake_run_shell_command(cmd, **kwargs):
    return cmd_map[' '.join(cmd)]