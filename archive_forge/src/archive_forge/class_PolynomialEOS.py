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
class PolynomialEOS(EOSBase):
    """
    Derives from EOSBase. Polynomial based equations of states must subclass
    this.
    """

    def _func(self, volume, params):
        return np.poly1d(list(params))(volume)

    def fit(self, order):
        """
        Do polynomial fitting and set the parameters. Uses numpy polyfit.

        Args:
            order (int): order of the fit polynomial
        """
        self.eos_params = np.polyfit(self.volumes, self.energies, order)
        self._set_params()

    def _set_params(self):
        """
        Use the fit polynomial to compute the parameter e0, b0, b1 and v0
        and set to the _params attribute.
        """
        fit_poly = np.poly1d(self.eos_params)
        v_e_min = self.volumes[np.argmin(self.energies)]
        min_wrt_v = minimize(fit_poly, v_e_min)
        e0, v0 = (min_wrt_v.fun, min_wrt_v.x[0])
        pderiv2 = np.polyder(fit_poly, 2)
        pderiv3 = np.polyder(fit_poly, 3)
        b0 = v0 * np.poly1d(pderiv2)(v0)
        db0dv = np.poly1d(pderiv2)(v0) + v0 * np.poly1d(pderiv3)(v0)
        b1 = -v0 * db0dv / b0
        self._params = [e0, b0, b1, v0]