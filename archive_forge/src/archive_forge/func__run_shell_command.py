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
def _run_shell_command(cmd, throw_on_error=False, buffer=True, env=None):
    if buffer:
        out_location = subprocess.PIPE
        err_location = subprocess.PIPE
    else:
        out_location = None
        err_location = None
    newenv = os.environ.copy()
    if env:
        newenv.update(env)
    output = subprocess.Popen(cmd, stdout=out_location, stderr=err_location, env=newenv)
    out = output.communicate()
    if output.returncode and throw_on_error:
        raise distutils.errors.DistutilsError('%s returned %d' % (cmd, output.returncode))
    if len(out) == 0 or not out[0] or (not out[0].strip()):
        return ''
    return out[0].strip().decode('utf-8', 'replace')