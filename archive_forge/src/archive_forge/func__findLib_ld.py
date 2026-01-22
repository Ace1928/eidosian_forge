import os
import shutil
import subprocess
import sys
def _findLib_ld(name):
    expr = '[^\\(\\)\\s]*lib%s\\.[^\\(\\)\\s]*' % re.escape(name)
    cmd = ['ld', '-t']
    libpath = os.environ.get('LD_LIBRARY_PATH')
    if libpath:
        for d in libpath.split(':'):
            cmd.extend(['-L', d])
    cmd.extend(['-o', os.devnull, '-l%s' % name])
    result = None
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        out, _ = p.communicate()
        res = re.findall(expr, os.fsdecode(out))
        for file in res:
            if not _is_elf(file):
                continue
            return os.fsdecode(file)
    except Exception:
        pass
    return result