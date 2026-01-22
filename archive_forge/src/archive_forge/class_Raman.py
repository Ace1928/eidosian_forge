import numpy as np
import ase.units as u
from ase.parallel import world
from ase.phonons import Phonons
from ase.vibrations.vibrations import Vibrations, AtomicDisplacements
from ase.dft import monkhorst_pack
from ase.utils import IOContext
class Raman(RamanData):

    def __init__(self, atoms, *args, **kwargs):
        super().__init__(atoms, *args, **kwargs)
        for key in ['txt', 'exext', 'exname']:
            kwargs.pop(key, None)
        kwargs['name'] = kwargs.get('name', self.name)
        self.vibrations = Vibrations(atoms, *args, **kwargs)
        self.delta = self.vibrations.delta
        self.indices = self.vibrations.indices

    def calculate_energies_and_modes(self):
        if hasattr(self, 'im_r'):
            return
        self.read()
        self.im_r = self.vibrations.im
        self.modes_Qq = self.vibrations.modes
        self.om_Q = self.vibrations.hnu.real
        self.H = self.vibrations.H
        with np.errstate(divide='ignore'):
            self.vib01_Q = np.where(self.om_Q > 0, 1.0 / np.sqrt(2 * self.om_Q), 0)
        self.vib01_Q *= np.sqrt(u.Ha * u._me / u._amu) * u.Bohr