from functools import partial
import warnings
import holoviews as _hv
import numpy as _np
from packaging.version import Version
from ..backend_transforms import _transfer_opts_cur_backend
from ..converter import HoloViewsConverter
from ..util import with_hv_extension, _convert_col_names_to_str

    Scatter matrix of numeric columns.

    A scatter_matrix shows all the pairwise relationships between the columns.
    Each non-diagonal plots the corresponding columns against each other,
    while the diagonal plot shows the distribution of each individual column.

    This function is closely modelled on :func:`pandas.plotting.scatter_matrix`.

    Parameters:
    -----------
    data: DataFrame
        The data to plot. Every column is compared to every other column.
    c: str, optional
        Column to color by
    chart: str, optional
        Chart type for the off-diagonal plots (one of 'scatter', 'bivariate', 'hexbin')
    diagonal: str, optional
        Chart type for the diagonal plots (one of 'hist', 'kde')
    alpha: float, optional
        Transparency level for the off-diagonal plots
    nonselection_alpha: float, optional
        Transparency level for nonselected object in the off-diagonal plots
    tools: list of str, optional
        Interaction tools to include
        Defaults are 'box_select' and 'lasso_select'
    cmap/colormap: str or colormap object, optional
        Colormap to use when ``c`` is set.
        Default is `Category10 <https://github.com/d3/d3-3.x-api-reference/blob/master/Ordinal-Scales.md#category10>`.
    diagonal_kwds/hist_kwds/density_kwds: dict, optional
        Keyword options for the diagonal plots
    datashade (default=False):
        Whether to apply rasterization and shading (colormapping) using
        the Datashader library, returning an RGB object instead of
        individual points
    rasterize (default=False):
        Whether to apply rasterization using the Datashader library,
        returning an aggregated Image (to be colormapped by the
        plotting backend) instead of individual points
    dynspread (default=False):
        For plots generated with datashade=True or rasterize=True,
        automatically increase the point size when the data is sparse
        so that individual points become more visible.
        kwds supported include ``max_px``, ``threshold``,  ``shape``, ``how`` and ``mask``.
    spread (default=False):
        Make plots generated with datashade=True or rasterize=True
        increase the point size to make points more visible, by
        applying a fixed spreading of a certain number of cells/pixels. kwds
        supported include: ``px``, ``shape``, ``how`` and ``mask``.
    kwds: Keyword options for the off-diagonal plots and datashader's spreading , optional

    Returns:
    --------
    obj : HoloViews object
        The HoloViews representation of the plot.

    See Also
    --------
        :func:`pandas.plotting.scatter_matrix` : Equivalent pandas function.
    