import os
import sys
import shutil
import time
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import jsonio
from ase.io.ulm import open as ulmopen
from ase.parallel import paropen, world, barrier
from ase.calculators.singlepoint import (SinglePointCalculator,
def _write_nframes(self, n):
    """Write the number of frames in the bundle."""
    assert self.state == 'write' or self.state == 'prewrite'
    with paropen(self.path / 'frames', 'w') as fd:
        fd.write(str(n) + '\n')