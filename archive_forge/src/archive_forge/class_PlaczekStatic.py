import numpy as np
import ase.units as u
from ase.vibrations.raman import Raman, RamanPhonons
from ase.vibrations.resonant_raman import ResonantRaman
from ase.calculators.excitation_list import polarizability
class PlaczekStatic(Raman):

    def read_excitations(self):
        """Read excitations from files written"""
        self.al0_rr = None
        self.alm_rr = []
        self.alp_rr = []
        for a, i in zip(self.myindices, self.myxyz):
            for sign, al_rr in zip([-1, 1], [self.alm_rr, self.alp_rr]):
                disp = self._disp(a, i, sign)
                al_rr.append(disp.load_static_polarizability())

    def electronic_me_Qcc(self):
        self.calculate_energies_and_modes()
        V_rcc = np.zeros((self.ndof, 3, 3), dtype=complex)
        pre = 1.0 / (2 * self.delta)
        pre *= u.Hartree * u.Bohr
        for i, r in enumerate(self.myr):
            V_rcc[r] = pre * (self.alp_rr[i] - self.alm_rr[i])
        self.comm.sum(V_rcc)
        return self.map_to_modes(V_rcc)