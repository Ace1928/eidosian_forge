import fcntl
import fnmatch
import glob
import json
import os
import plistlib
import re
import shutil
import struct
import subprocess
import sys
import tempfile
def ExecFilterLibtool(self, *cmd_list):
    """Calls libtool and filters out '/path/to/libtool: file: foo.o has no
    symbols'."""
    libtool_re = re.compile('^.*libtool: (?:for architecture: \\S* )?file: .* has no symbols$')
    libtool_re5 = re.compile('^.*libtool: warning for library: ' + '.* the table of contents is empty ' + '\\(no object file members in the library define global symbols\\)$')
    env = os.environ.copy()
    env['ZERO_AR_DATE'] = '1'
    libtoolout = subprocess.Popen(cmd_list, stderr=subprocess.PIPE, env=env)
    err = libtoolout.communicate()[1].decode('utf-8')
    for line in err.splitlines():
        if not libtool_re.match(line) and (not libtool_re5.match(line)):
            print(line, file=sys.stderr)
    if not libtoolout.returncode:
        for i in range(len(cmd_list) - 1):
            if cmd_list[i] == '-o' and cmd_list[i + 1].endswith('.a'):
                os.utime(cmd_list[i + 1], None)
                break
    return libtoolout.returncode