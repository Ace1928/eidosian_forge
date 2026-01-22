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
def _calc_recip(self):
    """
        Perform the reciprocal space summation. Calculates the quantity
        E_recip = 1/(2PiV) sum_{G < Gmax} exp(-(G.G/4/eta))/(G.G) S(G)S(-G)
        where
        S(G) = sum_{k=1,N} q_k exp(-i G.r_k)
        S(G)S(-G) = |S(G)|**2.

        This method is heavily vectorized to utilize numpy's C backend for speed.
        """
    n_sites = len(self._struct)
    prefactor = 2 * pi / self._vol
    e_recip = np.zeros((n_sites, n_sites), dtype=np.float64)
    forces = np.zeros((n_sites, 3), dtype=np.float64)
    coords = self._coords
    rcp_latt = self._struct.lattice.reciprocal_lattice
    recip_nn = rcp_latt.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], self._gmax)
    frac_coords = [frac_coords for frac_coords, dist, _idx, _img in recip_nn if dist != 0]
    gs = rcp_latt.get_cartesian_coords(frac_coords)
    g2s = np.sum(gs ** 2, 1)
    exp_vals = np.exp(-g2s / (4 * self._eta))
    grs = np.sum(gs[:, None] * coords[None, :], 2)
    oxi_states = np.array(self._oxi_states)
    qi_qj = oxi_states[None, :] * oxi_states[:, None]
    s_reals = np.sum(oxi_states[None, :] * np.cos(grs), 1)
    s_imags = np.sum(oxi_states[None, :] * np.sin(grs), 1)
    for g, g2, gr, exp_val, s_real, s_imag in zip(gs, g2s, grs, exp_vals, s_reals, s_imags):
        m = gr[None, :] + pi / 4 - gr[:, None]
        np.sin(m, m)
        m *= exp_val / g2
        e_recip += m
        if self._compute_forces:
            pref = 2 * exp_val / g2 * oxi_states
            factor = prefactor * pref * (s_real * np.sin(gr) - s_imag * np.cos(gr))
            forces += factor[:, None] * g[None, :]
    forces *= EwaldSummation.CONV_FACT
    e_recip *= prefactor * EwaldSummation.CONV_FACT * qi_qj * 2 ** 0.5
    return (e_recip, forces)