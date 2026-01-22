import os
import shutil
import subprocess
import sys
def _findLib_gcc(name):
    expr = os.fsencode('[^\\(\\)\\s]*lib%s\\.[^\\(\\)\\s]*' % re.escape(name))
    c_compiler = shutil.which('gcc')
    if not c_compiler:
        c_compiler = shutil.which('cc')
    if not c_compiler:
        return None
    temp = tempfile.NamedTemporaryFile()
    try:
        args = [c_compiler, '-Wl,-t', '-o', temp.name, '-l' + name]
        env = dict(os.environ)
        env['LC_ALL'] = 'C'
        env['LANG'] = 'C'
        try:
            proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        except OSError:
            return None
        with proc:
            trace = proc.stdout.read()
    finally:
        try:
            temp.close()
        except FileNotFoundError:
            pass
    res = re.findall(expr, trace)
    if not res:
        return None
    for file in res:
        if not _is_elf(file):
            continue
        return os.fsdecode(file)