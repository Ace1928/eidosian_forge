from __future__ import annotations
from statsmodels.compat.python import lrange
from collections import defaultdict
from io import StringIO
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.decorators import cache_readonly, deprecated_alias
from statsmodels.tools.linalg import logdet_symm
from statsmodels.tools.sm_exceptions import OutputWarning
from statsmodels.tools.validation import array_like
from statsmodels.tsa.base.tsa_model import (
import statsmodels.tsa.tsatools as tsa
from statsmodels.tsa.tsatools import duplication_matrix, unvec, vec
from statsmodels.tsa.vector_ar import output, plotting, util
from statsmodels.tsa.vector_ar.hypothesis_test_results import (
from statsmodels.tsa.vector_ar.irf import IRAnalysis
from statsmodels.tsa.vector_ar.output import VARSummary
class FEVD:
    """
    Compute and plot Forecast error variance decomposition and asymptotic
    standard errors
    """

    def __init__(self, model, P=None, periods=None):
        self.periods = periods
        self.model = model
        self.neqs = model.neqs
        self.names = model.model.endog_names
        self.irfobj = model.irf(var_decomp=P, periods=periods)
        self.orth_irfs = self.irfobj.orth_irfs
        irfs = (self.orth_irfs[:periods] ** 2).cumsum(axis=0)
        rng = lrange(self.neqs)
        mse = self.model.mse(periods)[:, rng, rng]
        fevd = np.empty_like(irfs)
        for i in range(periods):
            fevd[i] = (irfs[i].T / mse[i]).T
        self.decomp = fevd.swapaxes(0, 1)

    def summary(self):
        buf = StringIO()
        rng = lrange(self.periods)
        for i in range(self.neqs):
            ppm = output.pprint_matrix(self.decomp[i], rng, self.names)
            buf.write('FEVD for %s\n' % self.names[i])
            buf.write(ppm + '\n')
        print(buf.getvalue())

    def cov(self):
        """Compute asymptotic standard errors

        Returns
        -------
        """
        raise NotImplementedError

    def plot(self, periods=None, figsize=(10, 10), **plot_kwds):
        """Plot graphical display of FEVD

        Parameters
        ----------
        periods : int, default None
            Defaults to number originally specified. Can be at most that number
        """
        import matplotlib.pyplot as plt
        k = self.neqs
        periods = periods or self.periods
        fig, axes = plt.subplots(nrows=k, figsize=figsize)
        fig.suptitle('Forecast error variance decomposition (FEVD)')
        colors = [str(c) for c in np.arange(k, dtype=float) / k]
        ticks = np.arange(periods)
        limits = self.decomp.cumsum(2)
        ax = axes[0]
        for i in range(k):
            ax = axes[i]
            this_limits = limits[i].T
            handles = []
            for j in range(k):
                lower = this_limits[j - 1] if j > 0 else 0
                upper = this_limits[j]
                handle = ax.bar(ticks, upper - lower, bottom=lower, color=colors[j], label=self.names[j], **plot_kwds)
                handles.append(handle)
            ax.set_title(self.names[i])
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        plotting.adjust_subplots(right=0.85)
        return fig