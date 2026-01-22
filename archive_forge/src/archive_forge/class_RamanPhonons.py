import numpy as np
import ase.units as u
from ase.parallel import world
from ase.phonons import Phonons
from ase.vibrations.vibrations import Vibrations, AtomicDisplacements
from ase.dft import monkhorst_pack
from ase.utils import IOContext
class RamanPhonons(RamanData):

    def __init__(self, atoms, *args, **kwargs):
        RamanData.__init__(self, atoms, *args, **kwargs)
        for key in ['txt', 'exext', 'exname']:
            kwargs.pop(key, None)
        kwargs['name'] = kwargs.get('name', self.name)
        self.vibrations = Phonons(atoms, *args, **kwargs)
        self.delta = self.vibrations.delta
        self.indices = self.vibrations.indices
        self.kpts = (1, 1, 1)

    @property
    def kpts(self):
        return self._kpts

    @kpts.setter
    def kpts(self, kpts):
        if not hasattr(self, '_kpts') or kpts != self._kpts:
            self._kpts = kpts
            self.kpts_kc = monkhorst_pack(self.kpts)
            if hasattr(self, 'im_r'):
                del self.im_r

    def calculate_energies_and_modes(self):
        if not self._already_read:
            if hasattr(self, 'im_r'):
                del self.im_r
            self.read()
        if not hasattr(self, 'im_r'):
            omega_kl, u_kl = self.vibrations.band_structure(self.kpts_kc, modes=True, verbose=self.verbose)
            self.im_r = self.vibrations.m_inv_x
            self.om_Q = omega_kl.ravel().real
            self.modes_Qq = u_kl.reshape(len(self.om_Q), 3 * len(self.atoms))
            self.modes_Qq /= self.im_r
            self.om_v = self.om_Q
            with np.errstate(divide='ignore', invalid='ignore'):
                self.vib01_Q = np.where(self.om_Q > 0, 1.0 / np.sqrt(2 * self.om_Q), 0)
            self.vib01_Q *= np.sqrt(u.Ha * u._me / u._amu) * u.Bohr