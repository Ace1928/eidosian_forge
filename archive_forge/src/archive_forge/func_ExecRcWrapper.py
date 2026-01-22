import os
import re
import shutil
import subprocess
import stat
import string
import sys
def ExecRcWrapper(self, arch, *args):
    """Filter logo banner from invocations of rc.exe. Older versions of RC
    don't support the /nologo flag."""
    env = self._GetEnv(arch)
    popen = subprocess.Popen(args, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = popen.communicate()[0].decode('utf-8')
    for line in out.splitlines():
        if not line.startswith('Microsoft (R) Windows (R) Resource Compiler') and (not line.startswith('Copyright (C) Microsoft Corporation')) and line:
            print(line)
    return popen.returncode