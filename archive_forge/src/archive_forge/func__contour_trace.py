import plotly.colors as clrs
from plotly.graph_objs import graph_objs as go
from plotly import exceptions
from plotly import optional_imports
from skimage import measure
def _contour_trace(x, y, z, ncontours=None, colorscale='Electric', linecolor='rgb(150,150,150)', interp_mode='llr', coloring=None, v_min=0, v_max=1):
    """
    Contour trace in Cartesian coordinates.

    Parameters
    ==========

    x, y : array-like
        Cartesian coordinates
    z : array-like
        Field to be represented as contours.
    ncontours : int or None
        Number of contours to display (determined automatically if None).
    colorscale : None or str (Plotly colormap)
        colorscale of the contours.
    linecolor : rgb color
        Color used for lines. If ``colorscale`` is not None, line colors are
        determined from ``colorscale`` instead.
    interp_mode : 'ilr' (default) or 'cartesian'
        Defines how data are interpolated to compute contours. If 'irl',
        ILR (Isometric Log-Ratio) of compositional data is performed. If
        'cartesian', contours are determined in Cartesian space.
    coloring : None or 'lines'
        How to display contour. Filled contours if None, lines if ``lines``.
    vmin, vmax : float
        Bounds of interval of values used for the colorspace

    Notes
    =====
    """
    colors = _colors(ncontours + 2, colorscale)
    values = np.linspace(v_min, v_max, ncontours + 2)
    color_min, color_max = (colors[0], colors[-1])
    colors = colors[1:-1]
    values = values[1:-1]
    if linecolor is None:
        linecolor = 'rgb(150, 150, 150)'
    else:
        colors = [linecolor] * ncontours
    all_contours, all_values, all_areas, all_colors = _extract_contours(z, values, colors)
    order = np.argsort(all_areas)[::-1]
    all_contours, all_values, all_areas, all_colors, discrete_cm = _add_outer_contour(all_contours, all_values, all_areas, all_colors, values, all_values[order[0]], v_min, v_max, colors, color_min, color_max)
    order = np.concatenate(([0], order + 1))
    traces = []
    M, invM = _transform_barycentric_cartesian()
    dx = (x.max() - x.min()) / x.size
    dy = (y.max() - y.min()) / y.size
    for index in order:
        y_contour, x_contour = all_contours[index].T
        val = all_values[index]
        if interp_mode == 'cartesian':
            bar_coords = np.dot(invM, np.stack((dx * x_contour, dy * y_contour, np.ones(x_contour.shape))))
        elif interp_mode == 'ilr':
            bar_coords = _ilr_inverse(np.stack((dx * x_contour + x.min(), dy * y_contour + y.min())))
        if index == 0:
            a = np.array([1, 0, 0])
            b = np.array([0, 1, 0])
            c = np.array([0, 0, 1])
        else:
            a, b, c = bar_coords
        if _is_invalid_contour(x_contour, y_contour):
            continue
        _col = all_colors[index] if coloring == 'lines' else linecolor
        trace = dict(type='scatterternary', a=a, b=b, c=c, mode='lines', line=dict(color=_col, shape='spline', width=1), fill='toself', fillcolor=all_colors[index], showlegend=True, hoverinfo='skip', name='%.3f' % val)
        if coloring == 'lines':
            trace['fill'] = None
        traces.append(trace)
    return (traces, discrete_cm)