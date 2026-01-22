from math import pi, sqrt, log
import sys
import numpy as np
from pathlib import Path
import ase.units as units
import ase.io
from ase.parallel import world, paropen
from ase.utils.filecache import get_json_cache
from .data import VibrationsData
from collections import namedtuple
def _write_jmol(self, fd):
    symbols = self.atoms.get_chemical_symbols()
    freq = self.get_frequencies()
    for n in range(3 * len(self.indices)):
        fd.write('%6d\n' % len(self.atoms))
        if freq[n].imag != 0:
            c = 'i'
            freq[n] = freq[n].imag
        else:
            freq[n] = freq[n].real
            c = ' '
        fd.write('Mode #%d, f = %.1f%s cm^-1' % (n, float(freq[n].real), c))
        if self.ir:
            fd.write(', I = %.4f (D/Å)^2 amu^-1.\n' % self.intensities[n])
        else:
            fd.write('.\n')
        mode = self.get_mode(n)
        for i, pos in enumerate(self.atoms.positions):
            fd.write('%2s %12.5f %12.5f %12.5f %12.5f %12.5f %12.5f\n' % (symbols[i], pos[0], pos[1], pos[2], mode[i, 0], mode[i, 1], mode[i, 2]))