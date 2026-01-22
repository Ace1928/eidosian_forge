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
def _run_git_functions():
    git_dir = None
    if _git_is_installed():
        git_dir = _get_git_directory()
    return git_dir or None