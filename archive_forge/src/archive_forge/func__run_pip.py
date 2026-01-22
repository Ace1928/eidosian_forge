import collections
import os
import os.path
import subprocess
import sys
import sysconfig
import tempfile
from importlib import resources
import runpy
import sys
def _run_pip(args, additional_paths=None):
    code = f'\nimport runpy\nimport sys\nsys.path = {additional_paths or []} + sys.path\nsys.argv[1:] = {args}\nrunpy.run_module("pip", run_name="__main__", alter_sys=True)\n'
    cmd = [sys.executable, '-W', 'ignore::DeprecationWarning', '-c', code]
    if sys.flags.isolated:
        cmd.insert(1, '-I')
    return subprocess.run(cmd, check=True).returncode