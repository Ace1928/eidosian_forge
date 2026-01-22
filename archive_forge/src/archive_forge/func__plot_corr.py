from statsmodels.compat.pandas import deprecate_kwarg
import calendar
import numpy as np
import pandas as pd
from statsmodels.graphics import utils
from statsmodels.tools.validation import array_like
from statsmodels.tsa.stattools import acf, pacf, ccf
def _plot_corr(ax, title, acf_x, confint, lags, irregular, use_vlines, vlines_kwargs, auto_ylims=False, skip_lag0_confint=True, **kwargs):
    if irregular:
        acf_x = acf_x[lags]
        if confint is not None:
            confint = confint[lags]
    if use_vlines:
        ax.vlines(lags, [0], acf_x, **vlines_kwargs)
        ax.axhline(**kwargs)
    kwargs.setdefault('marker', 'o')
    kwargs.setdefault('markersize', 5)
    if 'ls' not in kwargs:
        kwargs.setdefault('linestyle', 'None')
    ax.margins(0.05)
    ax.plot(lags, acf_x, **kwargs)
    ax.set_title(title)
    ax.set_ylim(-1, 1)
    if auto_ylims:
        ax.set_ylim(1.25 * np.minimum(min(acf_x), min(confint[:, 0] - acf_x)), 1.25 * np.maximum(max(acf_x), max(confint[:, 1] - acf_x)))
    if confint is not None:
        if skip_lag0_confint and lags[0] == 0:
            lags = lags[1:]
            confint = confint[1:]
            acf_x = acf_x[1:]
        lags = lags.astype(float)
        lags[np.argmin(lags)] -= 0.5
        lags[np.argmax(lags)] += 0.5
        ax.fill_between(lags, confint[:, 0] - acf_x, confint[:, 1] - acf_x, alpha=0.25)