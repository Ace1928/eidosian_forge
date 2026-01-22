import copy
from textwrap import dedent
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from . import utils
from . import algorithms as algo
from .axisgrid import FacetGrid, _facet_docs
class _RegressionPlotter(_LinearPlotter):
    """Plotter for numeric independent variables with regression model.

    This does the computations and drawing for the `regplot` function, and
    is thus also used indirectly by `lmplot`.
    """

    def __init__(self, x, y, data=None, x_estimator=None, x_bins=None, x_ci='ci', scatter=True, fit_reg=True, ci=95, n_boot=1000, units=None, seed=None, order=1, logistic=False, lowess=False, robust=False, logx=False, x_partial=None, y_partial=None, truncate=False, dropna=True, x_jitter=None, y_jitter=None, color=None, label=None):
        self.x_estimator = x_estimator
        self.ci = ci
        self.x_ci = ci if x_ci == 'ci' else x_ci
        self.n_boot = n_boot
        self.seed = seed
        self.scatter = scatter
        self.fit_reg = fit_reg
        self.order = order
        self.logistic = logistic
        self.lowess = lowess
        self.robust = robust
        self.logx = logx
        self.truncate = truncate
        self.x_jitter = x_jitter
        self.y_jitter = y_jitter
        self.color = color
        self.label = label
        if sum((order > 1, logistic, robust, lowess, logx)) > 1:
            raise ValueError('Mutually exclusive regression options.')
        self.establish_variables(data, x=x, y=y, units=units, x_partial=x_partial, y_partial=y_partial)
        if dropna:
            self.dropna('x', 'y', 'units', 'x_partial', 'y_partial')
        if self.x_partial is not None:
            self.x = self.regress_out(self.x, self.x_partial)
        if self.y_partial is not None:
            self.y = self.regress_out(self.y, self.y_partial)
        if x_bins is not None:
            self.x_estimator = np.mean if x_estimator is None else x_estimator
            x_discrete, x_bins = self.bin_predictor(x_bins)
            self.x_discrete = x_discrete
        else:
            self.x_discrete = self.x
        if len(self.x) <= 1:
            self.fit_reg = False
        if self.fit_reg:
            self.x_range = (self.x.min(), self.x.max())

    @property
    def scatter_data(self):
        """Data where each observation is a point."""
        x_j = self.x_jitter
        if x_j is None:
            x = self.x
        else:
            x = self.x + np.random.uniform(-x_j, x_j, len(self.x))
        y_j = self.y_jitter
        if y_j is None:
            y = self.y
        else:
            y = self.y + np.random.uniform(-y_j, y_j, len(self.y))
        return (x, y)

    @property
    def estimate_data(self):
        """Data with a point estimate and CI for each discrete x value."""
        x, y = (self.x_discrete, self.y)
        vals = sorted(np.unique(x))
        points, cis = ([], [])
        for val in vals:
            _y = y[x == val]
            est = self.x_estimator(_y)
            points.append(est)
            if self.x_ci is None:
                cis.append(None)
            else:
                units = None
                if self.x_ci == 'sd':
                    sd = np.std(_y)
                    _ci = (est - sd, est + sd)
                else:
                    if self.units is not None:
                        units = self.units[x == val]
                    boots = algo.bootstrap(_y, func=self.x_estimator, n_boot=self.n_boot, units=units, seed=self.seed)
                    _ci = utils.ci(boots, self.x_ci)
                cis.append(_ci)
        return (vals, points, cis)

    def _check_statsmodels(self):
        """Check whether statsmodels is installed if any boolean options require it."""
        options = ('logistic', 'robust', 'lowess')
        err = '`{}=True` requires statsmodels, an optional dependency, to be installed.'
        for option in options:
            if getattr(self, option) and (not _has_statsmodels):
                raise RuntimeError(err.format(option))

    def fit_regression(self, ax=None, x_range=None, grid=None):
        """Fit the regression model."""
        self._check_statsmodels()
        if grid is None:
            if self.truncate:
                x_min, x_max = self.x_range
            elif ax is None:
                x_min, x_max = x_range
            else:
                x_min, x_max = ax.get_xlim()
            grid = np.linspace(x_min, x_max, 100)
        ci = self.ci
        if self.order > 1:
            yhat, yhat_boots = self.fit_poly(grid, self.order)
        elif self.logistic:
            from statsmodels.genmod.generalized_linear_model import GLM
            from statsmodels.genmod.families import Binomial
            yhat, yhat_boots = self.fit_statsmodels(grid, GLM, family=Binomial())
        elif self.lowess:
            ci = None
            grid, yhat = self.fit_lowess()
        elif self.robust:
            from statsmodels.robust.robust_linear_model import RLM
            yhat, yhat_boots = self.fit_statsmodels(grid, RLM)
        elif self.logx:
            yhat, yhat_boots = self.fit_logx(grid)
        else:
            yhat, yhat_boots = self.fit_fast(grid)
        if ci is None:
            err_bands = None
        else:
            err_bands = utils.ci(yhat_boots, ci, axis=0)
        return (grid, yhat, err_bands)

    def fit_fast(self, grid):
        """Low-level regression and prediction using linear algebra."""

        def reg_func(_x, _y):
            return np.linalg.pinv(_x).dot(_y)
        X, y = (np.c_[np.ones(len(self.x)), self.x], self.y)
        grid = np.c_[np.ones(len(grid)), grid]
        yhat = grid.dot(reg_func(X, y))
        if self.ci is None:
            return (yhat, None)
        beta_boots = algo.bootstrap(X, y, func=reg_func, n_boot=self.n_boot, units=self.units, seed=self.seed).T
        yhat_boots = grid.dot(beta_boots).T
        return (yhat, yhat_boots)

    def fit_poly(self, grid, order):
        """Regression using numpy polyfit for higher-order trends."""

        def reg_func(_x, _y):
            return np.polyval(np.polyfit(_x, _y, order), grid)
        x, y = (self.x, self.y)
        yhat = reg_func(x, y)
        if self.ci is None:
            return (yhat, None)
        yhat_boots = algo.bootstrap(x, y, func=reg_func, n_boot=self.n_boot, units=self.units, seed=self.seed)
        return (yhat, yhat_boots)

    def fit_statsmodels(self, grid, model, **kwargs):
        """More general regression function using statsmodels objects."""
        import statsmodels.tools.sm_exceptions as sme
        X, y = (np.c_[np.ones(len(self.x)), self.x], self.y)
        grid = np.c_[np.ones(len(grid)), grid]

        def reg_func(_x, _y):
            err_classes = (sme.PerfectSeparationError,)
            try:
                with warnings.catch_warnings():
                    if hasattr(sme, 'PerfectSeparationWarning'):
                        warnings.simplefilter('error', sme.PerfectSeparationWarning)
                        err_classes = (*err_classes, sme.PerfectSeparationWarning)
                    yhat = model(_y, _x, **kwargs).fit().predict(grid)
            except err_classes:
                yhat = np.empty(len(grid))
                yhat.fill(np.nan)
            return yhat
        yhat = reg_func(X, y)
        if self.ci is None:
            return (yhat, None)
        yhat_boots = algo.bootstrap(X, y, func=reg_func, n_boot=self.n_boot, units=self.units, seed=self.seed)
        return (yhat, yhat_boots)

    def fit_lowess(self):
        """Fit a locally-weighted regression, which returns its own grid."""
        from statsmodels.nonparametric.smoothers_lowess import lowess
        grid, yhat = lowess(self.y, self.x).T
        return (grid, yhat)

    def fit_logx(self, grid):
        """Fit the model in log-space."""
        X, y = (np.c_[np.ones(len(self.x)), self.x], self.y)
        grid = np.c_[np.ones(len(grid)), np.log(grid)]

        def reg_func(_x, _y):
            _x = np.c_[_x[:, 0], np.log(_x[:, 1])]
            return np.linalg.pinv(_x).dot(_y)
        yhat = grid.dot(reg_func(X, y))
        if self.ci is None:
            return (yhat, None)
        beta_boots = algo.bootstrap(X, y, func=reg_func, n_boot=self.n_boot, units=self.units, seed=self.seed).T
        yhat_boots = grid.dot(beta_boots).T
        return (yhat, yhat_boots)

    def bin_predictor(self, bins):
        """Discretize a predictor by assigning value to closest bin."""
        x = np.asarray(self.x)
        if np.isscalar(bins):
            percentiles = np.linspace(0, 100, bins + 2)[1:-1]
            bins = np.percentile(x, percentiles)
        else:
            bins = np.ravel(bins)
        dist = np.abs(np.subtract.outer(x, bins))
        x_binned = bins[np.argmin(dist, axis=1)].ravel()
        return (x_binned, bins)

    def regress_out(self, a, b):
        """Regress b from a keeping a's original mean."""
        a_mean = a.mean()
        a = a - a_mean
        b = b - b.mean()
        b = np.c_[b]
        a_prime = a - b.dot(np.linalg.pinv(b).dot(a))
        return np.asarray(a_prime + a_mean).reshape(a.shape)

    def plot(self, ax, scatter_kws, line_kws):
        """Draw the full plot."""
        if self.scatter:
            scatter_kws['label'] = self.label
        else:
            line_kws['label'] = self.label
        if self.color is None:
            lines, = ax.plot([], [])
            color = lines.get_color()
            lines.remove()
        else:
            color = self.color
        color = mpl.colors.rgb2hex(mpl.colors.colorConverter.to_rgb(color))
        scatter_kws.setdefault('color', color)
        line_kws.setdefault('color', color)
        if self.scatter:
            self.scatterplot(ax, scatter_kws)
        if self.fit_reg:
            self.lineplot(ax, line_kws)
        if hasattr(self.x, 'name'):
            ax.set_xlabel(self.x.name)
        if hasattr(self.y, 'name'):
            ax.set_ylabel(self.y.name)

    def scatterplot(self, ax, kws):
        """Draw the data."""
        line_markers = ['1', '2', '3', '4', '+', 'x', '|', '_']
        if self.x_estimator is None:
            if 'marker' in kws and kws['marker'] in line_markers:
                lw = mpl.rcParams['lines.linewidth']
            else:
                lw = mpl.rcParams['lines.markeredgewidth']
            kws.setdefault('linewidths', lw)
            if not hasattr(kws['color'], 'shape') or kws['color'].shape[1] < 4:
                kws.setdefault('alpha', 0.8)
            x, y = self.scatter_data
            ax.scatter(x, y, **kws)
        else:
            ci_kws = {'color': kws['color']}
            if 'alpha' in kws:
                ci_kws['alpha'] = kws['alpha']
            ci_kws['linewidth'] = mpl.rcParams['lines.linewidth'] * 1.75
            kws.setdefault('s', 50)
            xs, ys, cis = self.estimate_data
            if [ci for ci in cis if ci is not None]:
                for x, ci in zip(xs, cis):
                    ax.plot([x, x], ci, **ci_kws)
            ax.scatter(xs, ys, **kws)

    def lineplot(self, ax, kws):
        """Draw the model."""
        grid, yhat, err_bands = self.fit_regression(ax)
        edges = (grid[0], grid[-1])
        fill_color = kws['color']
        lw = kws.pop('lw', mpl.rcParams['lines.linewidth'] * 1.5)
        kws.setdefault('linewidth', lw)
        line, = ax.plot(grid, yhat, **kws)
        if not self.truncate:
            line.sticky_edges.x[:] = edges
        if err_bands is not None:
            ax.fill_between(grid, *err_bands, facecolor=fill_color, alpha=0.15)