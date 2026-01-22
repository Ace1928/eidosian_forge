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
def ExecCopyInfoPlist(self, source, dest, convert_to_binary, *keys):
    """Copies the |source| Info.plist to the destination directory |dest|."""
    with open(source) as fd:
        lines = fd.read()
    plist = plistlib.readPlistFromString(lines)
    if keys:
        plist.update(json.loads(keys[0]))
    lines = plistlib.writePlistToString(plist)
    IDENT_RE = re.compile('[_/\\s]')
    for key in os.environ:
        if key.startswith('_'):
            continue
        evar = '${%s}' % key
        evalue = os.environ[key]
        lines = lines.replace(lines, evar, evalue)
        evar = '${%s:identifier}' % key
        evalue = IDENT_RE.sub('_', os.environ[key])
        lines = lines.replace(lines, evar, evalue)
        evar = '${%s:rfc1034identifier}' % key
        evalue = IDENT_RE.sub('-', os.environ[key])
        lines = lines.replace(lines, evar, evalue)
    lines = lines.splitlines()
    for i in range(len(lines)):
        if lines[i].strip().startswith('<string>${'):
            lines[i] = None
            lines[i - 1] = None
    lines = '\n'.join((line for line in lines if line is not None))
    with open(dest, 'w') as fd:
        fd.write(lines)
    self._WritePkgInfo(dest)
    if convert_to_binary == 'True':
        self._ConvertToBinary(dest)