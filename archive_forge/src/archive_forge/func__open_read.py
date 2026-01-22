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
def _open_read(self):
    """Open a bundle trajectory for reading."""
    if not os.path.exists(self.filename):
        raise IOError('File not found: ' + self.filename)
    if not self.is_bundle(self.filename):
        raise IOError('Not a BundleTrajectory: ' + self.filename)
    self.state = 'read'
    metadata = self._read_metadata()
    self.metadata = metadata
    if metadata['version'] > self.version:
        raise NotImplementedError('This version of ASE cannot read a BundleTrajectory version ' + str(metadata['version']))
    if metadata['subtype'] not in ('normal', 'split'):
        raise NotImplementedError('This version of ASE cannot read BundleTrajectory subtype ' + metadata['subtype'])
    self.subtype = metadata['subtype']
    if metadata['backend'] == 'ulm':
        self.singleprecision = metadata['ulm.singleprecision']
    self._set_backend(metadata['backend'])
    self.nframes = self._read_nframes()
    if self.nframes == 0:
        raise IOError('Empty BundleTrajectory')
    self.datatypes = metadata['datatypes']
    try:
        self.pythonmajor = metadata['python_ver'][0]
    except KeyError:
        self.pythonmajor = 2
    self.backend.readpy2 = self.pythonmajor == 2
    self.state = 'read'