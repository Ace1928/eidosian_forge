import sys
import numpy as np
from itertools import combinations_with_replacement
import ase.units as u
from ase.parallel import parprint, paropen
from ase.vibrations.resonant_raman import ResonantRaman
from ase.vibrations.franck_condon import FranckCondonOverlap
from ase.vibrations.franck_condon import FranckCondonRecursive
def unitless_displacements(self, forces_r, mineigv=1e-12):
    """Evaluate unitless displacements from forces

        Parameters
        ----------
        forces_r: array
          Forces in cartesian coordinates
        mineigv: float
          Minimal Eigenvalue to consider in matrix inversion to handle
          numerical noise. Default 1e-12

        Returns
        -------
        Unitless displacements in Eigenmode coordinates
        """
    assert len(forces_r.flat) == self.ndof
    if not hasattr(self, 'Dm1_q'):
        self.eigv_q, self.eigw_rq = np.linalg.eigh(self.im_r[:, None] * self.H * self.im_r)
        self.Dm1_q = np.divide(1, self.eigv_q, out=np.zeros_like(self.eigv_q), where=np.abs(self.eigv_q) > mineigv)
    X_r = self.eigw_rq @ np.diag(self.Dm1_q) @ self.eigw_rq.T @ (forces_r.flat * self.im_r)
    d_Q = np.dot(self.modes_Qq, X_r)
    s = 1e-20 / u.kg / u.C / u._hbar ** 2
    d_Q *= np.sqrt(s * self.om_Q)
    return d_Q