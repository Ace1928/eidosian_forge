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
def write_dos(self, out='vib-dos.dat', start=800, end=4000, npts=None, width=10, type='Gaussian', method='standard', direction='central'):
    """Write out the vibrational density of states to file.

        First column is the wavenumber in cm^-1, the second column the
        folded vibrational density of states.
        Start and end points, and width of the Gaussian/Lorentzian
        should be given in cm^-1."""
    frequencies = self.get_frequencies(method, direction).real
    intensities = np.ones(len(frequencies))
    energies, spectrum = self.fold(frequencies, intensities, start, end, npts, width, type)
    outdata = np.empty([len(energies), 2])
    outdata.T[0] = energies
    outdata.T[1] = spectrum
    with open(out, 'w') as fd:
        fd.write('# %s folded, width=%g cm^-1\n' % (type.title(), width))
        fd.write('# [cm^-1] arbitrary\n')
        for row in outdata:
            fd.write('%.3f  %15.5e\n' % (row[0], row[1]))