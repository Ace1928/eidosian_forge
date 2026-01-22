from io import BytesIO
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from contextlib import contextmanager
from ase.io.formats import ioformats
from ase.io import write
def _pipe_to_ase_gui(atoms, repeat):
    buf = BytesIO()
    write(buf, atoms, format='traj')
    args = [sys.executable, '-m', 'ase', 'gui', '-']
    if repeat:
        args.append('--repeat={},{},{}'.format(*repeat))
    proc = subprocess.Popen(args, stdin=subprocess.PIPE)
    proc.stdin.write(buf.getvalue())
    proc.stdin.close()
    return proc