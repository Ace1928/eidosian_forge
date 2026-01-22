import functools
import gc
import os
import platform
import re
import socket
import subprocess
import sys
import time
def _assert_python(expected_success, *args, **env_vars):
    if '__isolated' in env_vars:
        isolated = env_vars.pop('__isolated')
    else:
        isolated = not env_vars
    cmd_line = [sys.executable, '-X', 'faulthandler']
    if isolated and sys.version_info >= (3, 4):
        cmd_line.append('-I')
    elif not env_vars:
        cmd_line.append('-E')
    env = os.environ.copy()
    if env_vars.pop('__cleanenv', None):
        env = {}
    env.update(env_vars)
    cmd_line.extend(args)
    p = subprocess.Popen(cmd_line, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    try:
        out, err = p.communicate()
    finally:
        subprocess._cleanup()
        p.stdout.close()
        p.stderr.close()
    rc = p.returncode
    err = strip_python_stderr(err)
    if rc and expected_success or (not rc and (not expected_success)):
        raise AssertionError('Process return code is %d, stderr follows:\n%s' % (rc, err.decode('ascii', 'ignore')))
    return (rc, out, err)