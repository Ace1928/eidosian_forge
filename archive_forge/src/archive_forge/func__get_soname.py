import os
import shutil
import subprocess
import sys
def _get_soname(f):
    if not f:
        return None
    objdump = shutil.which('objdump')
    if not objdump:
        return None
    try:
        proc = subprocess.Popen((objdump, '-p', '-j', '.dynamic', f), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError:
        return None
    with proc:
        dump = proc.stdout.read()
    res = re.search(b'\\sSONAME\\s+([^\\s]+)', dump)
    if not res:
        return None
    return os.fsdecode(res.group(1))