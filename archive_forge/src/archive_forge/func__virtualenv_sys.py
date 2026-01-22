from __future__ import with_statement
import logging
import optparse
import os
import os.path
import re
import shutil
import subprocess
import sys
import itertools
def _virtualenv_sys(venv_path):
    """obtain version and path info from a virtualenv."""
    executable = os.path.join(venv_path, env_bin_dir, 'python')
    if _WIN32:
        env = os.environ.copy()
    else:
        env = {}
    p = subprocess.Popen([executable, '-c', 'import sys;print ("%d.%d" % (sys.version_info.major, sys.version_info.minor));print ("\\n".join(sys.path));'], env=env, stdout=subprocess.PIPE)
    stdout, err = p.communicate()
    assert not p.returncode and stdout
    lines = stdout.decode('utf-8').splitlines()
    return (lines[0], list(filter(bool, lines[1:])))