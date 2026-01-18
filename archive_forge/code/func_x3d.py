from io import BytesIO
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from contextlib import contextmanager
from ase.io.formats import ioformats
from ase.io import write
def x3d(self, atoms):
    from ase.visualize.x3d import view_x3d
    return view_x3d(atoms)