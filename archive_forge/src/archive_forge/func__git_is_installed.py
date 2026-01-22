from __future__ import unicode_literals
import distutils.errors
from distutils import log
import errno
import io
import os
import re
import subprocess
import time
import pkg_resources
from pbr import options
from pbr import version
def _git_is_installed():
    try:
        _run_shell_command(['git', '--version'])
    except OSError:
        return False
    return True