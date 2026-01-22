from __future__ import annotations
import logging
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING
import numpy as np
from scipy.optimize import leastsq, minimize
from pymatgen.core.units import FloatWithUnit
from pymatgen.util.plotting import add_fig_kwargs, get_ax_fig, pretty_plot
class NumericalEOS(PolynomialEOS):
    """A numerical EOS."""

    def fit(self, min_ndata_factor=3, max_poly_order_factor=5, min_poly_order=2):
        """
        Fit the input data to the 'numerical eos', the equation of state employed
        in the quasiharmonic Debye model described in the paper:
        10.1103/PhysRevB.90.174107.

        credits: Cormac Toher

        Args:
            min_ndata_factor (int): parameter that controls the minimum number
                of data points that will be used for fitting.
                minimum number of data points = total data points-2*min_ndata_factor
            max_poly_order_factor (int): parameter that limits the max order
                of the polynomial used for fitting.
                max_poly_order = number of data points used for fitting -
                max_poly_order_factor
            min_poly_order (int): minimum order of the polynomial to be
                considered for fitting.
        """
        warnings.simplefilter('ignore', np.RankWarning)

        def get_rms(x, y):
            return np.sqrt(np.sum((np.array(x) - np.array(y)) ** 2) / len(x))
        e_v = list(zip(self.energies, self.volumes))
        n_data = len(e_v)
        n_data_min = max(n_data - 2 * min_ndata_factor, min_poly_order + 1)
        rms_min = np.inf
        n_data_fit = n_data
        all_coeffs = {}
        e_v = sorted(e_v, key=lambda x: x[0])
        e_min = e_v[0]
        e_v = sorted(e_v, key=lambda x: x[1])
        e_min_idx = e_v.index(e_min)
        v_before = e_v[e_min_idx - 1][1]
        v_after = e_v[e_min_idx + 1][1]
        e_v_work = deepcopy(e_v)
        while n_data_fit >= n_data_min and e_min in e_v_work:
            max_poly_order = n_data_fit - max_poly_order_factor
            energies = [ei[0] for ei in e_v_work]
            volumes = [ei[1] for ei in e_v_work]
            for idx in range(min_poly_order, max_poly_order + 1):
                coeffs = np.polyfit(volumes, energies, idx)
                polyder = np.polyder(coeffs)
                a = np.poly1d(polyder)(v_before)
                b = np.poly1d(polyder)(v_after)
                if a * b < 0:
                    rms = get_rms(energies, np.poly1d(coeffs)(volumes))
                    rms_min = min(rms_min, rms * idx / n_data_fit)
                    all_coeffs[idx, n_data_fit] = [coeffs.tolist(), rms]
                    all_coeffs[idx, n_data_fit][0].reverse()
            e_v_work.pop()
            e_v_work.pop(0)
            n_data_fit = len(e_v_work)
        logger.info(f'total number of polynomials: {len(all_coeffs)}')
        norm = 0.0
        fit_poly_order = n_data
        weighted_avg_coeffs = np.zeros((fit_poly_order,))
        for key, val in all_coeffs.items():
            weighted_rms = val[1] * key[0] / rms_min / key[1]
            weight = np.exp(-weighted_rms ** 2)
            norm += weight
            coeffs = np.array(val[0])
            coeffs = np.lib.pad(coeffs, (0, max(fit_poly_order - len(coeffs), 0)), 'constant')
            weighted_avg_coeffs += weight * coeffs
        weighted_avg_coeffs /= norm
        weighted_avg_coeffs = weighted_avg_coeffs.tolist()
        weighted_avg_coeffs.reverse()
        self.eos_params = weighted_avg_coeffs
        self._set_params()