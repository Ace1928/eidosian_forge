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
class Murnaghan(EOSBase):
    """Murnaghan EOS."""

    def _func(self, volume, params):
        """From PRB 28,5480 (1983)."""
        e0, b0, b1, v0 = tuple(params)
        return e0 + b0 * volume / b1 * ((v0 / volume) ** b1 / (b1 - 1.0) + 1.0) - v0 * b0 / (b1 - 1.0)