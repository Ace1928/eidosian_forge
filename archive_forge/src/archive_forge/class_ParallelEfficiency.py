from __future__ import annotations
import collections
import logging
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from pymatgen.io.core import ParseError
from pymatgen.util.plotting import add_fig_kwargs, get_ax_fig
class ParallelEfficiency(dict):
    """Store results concerning the parallel efficiency of the job."""

    def __init__(self, filenames, ref_idx, *args, **kwargs):
        """
        Args:
            filenames: List of filenames
            ref_idx: Index of the Reference time (calculation done with the smallest number of cpus).
        """
        self.update(*args, **kwargs)
        self.filenames = filenames
        self._ref_idx = ref_idx

    def _order_by_peff(self, key, criterion, reverse=True):
        self.estimator = {'min': min, 'max': max, 'mean': lambda items: sum(items) / len(items)}[criterion]
        data = []
        for sect_name, peff in self.items():
            if all((v != -1 for v in peff[key])):
                values = peff[key][:]
                if len(values) > 1:
                    ref_value = values.pop(self._ref_idx)
                    assert ref_value == 1.0
                data.append((sect_name, self.estimator(values)))
        data.sort(key=lambda t: t[1], reverse=reverse)
        return tuple((sect_name for sect_name, e in data))

    def totable(self, stop=None, reverse=True):
        """
        Return table (list of lists) with timing results.

        Args:
            stop: Include results up to stop. None for all
            reverse: Put items with highest wall_time in first positions if True.
        """
        osects = self._order_by_peff('wall_time', criterion='mean', reverse=reverse)
        if stop is not None:
            osects = osects[:stop]
        n = len(self.filenames)
        table = [['AbinitTimerSection', *alternate(self.filenames, n * ['%'])]]
        for sect_name in osects:
            peff = self[sect_name]['wall_time']
            fract = self[sect_name]['wall_fract']
            vals = alternate(peff, fract)
            table.append([sect_name] + [f'{val:.2f}' for val in vals])
        return table

    def good_sections(self, key='wall_time', criterion='mean', nmax=5):
        """Return first `nmax` sections with best value of key `key` using criterion `criterion`."""
        good_sections = self._order_by_peff(key, criterion=criterion)
        return good_sections[:nmax]

    def bad_sections(self, key='wall_time', criterion='mean', nmax=5):
        """Return first `nmax` sections with worst value of key `key` using criterion `criterion`."""
        bad_sections = self._order_by_peff(key, criterion=criterion, reverse=False)
        return bad_sections[:nmax]