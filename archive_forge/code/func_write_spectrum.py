import sys
import numpy as np
import ase.units as u
from ase.parallel import world, paropen, parprint
from ase.vibrations import Vibrations
from ase.vibrations.raman import Raman, RamanCalculatorBase
def write_spectrum(self, omega, gamma, out='resonant-raman-spectra.dat', start=200, end=4000, npts=None, width=10, type='Gaussian'):
    """Write out spectrum to file.

        Start and end
        point, and width of the Gaussian/Lorentzian should be given
        in cm^-1."""
    energies, spectrum = self.get_spectrum(omega, gamma, start, end, npts, width, type)
    outdata = np.empty([len(energies), 3])
    outdata.T[0] = energies
    outdata.T[1] = spectrum
    with paropen(out, 'w') as fd:
        fd.write('# Resonant Raman spectrum\n')
        if hasattr(self, '_approx'):
            fd.write('# approximation: {0}\n'.format(self._approx))
        for key in self.observation:
            fd.write('# {0}: {1}\n'.format(key, self.observation[key]))
        fd.write('# omega={0:g} eV, gamma={1:g} eV\n'.format(omega, gamma))
        if width is not None:
            fd.write('# %s folded, width=%g cm^-1\n' % (type.title(), width))
        fd.write('# [cm^-1]  [a.u.]\n')
        for row in outdata:
            fd.write('%.3f  %15.5g\n' % (row[0], row[1]))