from __future__ import annotations
import bisect
from copy import copy, deepcopy
from datetime import datetime
from math import log, pi, sqrt
from typing import TYPE_CHECKING, Any
from warnings import warn
import numpy as np
from monty.json import MSONable
from scipy import constants
from scipy.special import comb, erfc
from pymatgen.core.structure import Structure
from pymatgen.util.due import Doi, due
def _calc_real_and_point(self):
    """Determines the self energy -(eta/pi)**(1/2) * sum_{i=1}^{N} q_i**2."""
    frac_coords = self._struct.frac_coords
    force_pf = 2 * self._sqrt_eta / sqrt(pi)
    coords = self._coords
    n_sites = len(self._struct)
    e_real = np.empty((n_sites, n_sites), dtype=np.float64)
    forces = np.zeros((n_sites, 3), dtype=np.float64)
    qs = np.array(self._oxi_states)
    e_point = -qs ** 2 * sqrt(self._eta / pi)
    for idx in range(n_sites):
        nf_coords, rij, js, _ = self._struct.lattice.get_points_in_sphere(frac_coords, coords[idx], self._rmax, zip_results=False)
        inds = rij > 1e-08
        js = js[inds]
        rij = rij[inds]
        nf_coords = nf_coords[inds]
        qi = qs[idx]
        qj = qs[js]
        erfc_val = erfc(self._sqrt_eta * rij)
        new_ereals = erfc_val * qi * qj / rij
        for key in range(n_sites):
            e_real[key, idx] = np.sum(new_ereals[js == key])
        if self._compute_forces:
            nc_coords = self._struct.lattice.get_cartesian_coords(nf_coords)
            fijpf = qj / rij ** 3 * (erfc_val + force_pf * rij * np.exp(-self._eta * rij ** 2))
            forces[idx] += np.sum(np.expand_dims(fijpf, 1) * (np.array([coords[idx]]) - nc_coords) * qi * EwaldSummation.CONV_FACT, axis=0)
    e_real *= 0.5 * EwaldSummation.CONV_FACT
    e_point *= EwaldSummation.CONV_FACT
    return (e_real, e_point, forces)