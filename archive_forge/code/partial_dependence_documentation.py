import numbers
from itertools import chain
from math import ceil
import numpy as np
from scipy import sparse
from scipy.stats.mstats import mquantiles
from ...base import is_regressor
from ...utils import (
from ...utils._encode import _unique
from ...utils.parallel import Parallel, delayed
from .. import partial_dependence
from .._pd_utils import _check_feature_names, _get_feature_index
Plot partial dependence plots.

        Parameters
        ----------
        ax : Matplotlib axes or array-like of Matplotlib axes, default=None
            - If a single axis is passed in, it is treated as a bounding axes
                and a grid of partial dependence plots will be drawn within
                these bounds. The `n_cols` parameter controls the number of
                columns in the grid.
            - If an array-like of axes are passed in, the partial dependence
                plots will be drawn directly into these axes.
            - If `None`, a figure and a bounding axes is created and treated
                as the single axes case.

        n_cols : int, default=3
            The maximum number of columns in the grid plot. Only active when
            `ax` is a single axes or `None`.

        line_kw : dict, default=None
            Dict with keywords passed to the `matplotlib.pyplot.plot` call.
            For one-way partial dependence plots.

        ice_lines_kw : dict, default=None
            Dictionary with keywords passed to the `matplotlib.pyplot.plot` call.
            For ICE lines in the one-way partial dependence plots.
            The key value pairs defined in `ice_lines_kw` takes priority over
            `line_kw`.

            .. versionadded:: 1.0

        pd_line_kw : dict, default=None
            Dictionary with keywords passed to the `matplotlib.pyplot.plot` call.
            For partial dependence in one-way partial dependence plots.
            The key value pairs defined in `pd_line_kw` takes priority over
            `line_kw`.

            .. versionadded:: 1.0

        contour_kw : dict, default=None
            Dict with keywords passed to the `matplotlib.pyplot.contourf`
            call for two-way partial dependence plots.

        bar_kw : dict, default=None
            Dict with keywords passed to the `matplotlib.pyplot.bar`
            call for one-way categorical partial dependence plots.

            .. versionadded:: 1.2

        heatmap_kw : dict, default=None
            Dict with keywords passed to the `matplotlib.pyplot.imshow`
            call for two-way categorical partial dependence plots.

            .. versionadded:: 1.2

        pdp_lim : dict, default=None
            Global min and max average predictions, such that all plots will have the
            same scale and y limits. `pdp_lim[1]` is the global min and max for single
            partial dependence curves. `pdp_lim[2]` is the global min and max for
            two-way partial dependence curves. If `None` (default), the limit will be
            inferred from the global minimum and maximum of all predictions.

            .. versionadded:: 1.1

        centered : bool, default=False
            If `True`, the ICE and PD lines will start at the origin of the
            y-axis. By default, no centering is done.

            .. versionadded:: 1.1

        Returns
        -------
        display : :class:`~sklearn.inspection.PartialDependenceDisplay`
            Returns a :class:`~sklearn.inspection.PartialDependenceDisplay`
            object that contains the partial dependence plots.
        