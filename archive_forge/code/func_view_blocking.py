from io import BytesIO
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from contextlib import contextmanager
from ase.io.formats import ioformats
from ase.io import write
def view_blocking(self, atoms, data=None):
    with self.mktemp(atoms, data) as path:
        subprocess.check_call(self.argv + [str(path)])