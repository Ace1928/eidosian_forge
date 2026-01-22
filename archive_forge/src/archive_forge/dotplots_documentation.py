import numpy as np
from . import utils

    Dot plotting (also known as forest and blobbogram).

    Produce a dotplot similar in style to those in Cleveland's
    "Visualizing Data" book ([1]_).  These are also known as "forest plots".

    Parameters
    ----------
    points : array_like
        The quantitative values to be plotted as markers.
    intervals : array_like
        The intervals to be plotted around the points.  The elements
        of `intervals` are either scalars or sequences of length 2.  A
        scalar indicates the half width of a symmetric interval.  A
        sequence of length 2 contains the left and right half-widths
        (respectively) of a nonsymmetric interval.  If None, no
        intervals are drawn.
    lines : array_like
        A grouping variable indicating which points/intervals are
        drawn on a common line.  If None, each point/interval appears
        on its own line.
    sections : array_like
        A grouping variable indicating which lines are grouped into
        sections.  If None, everything is drawn in a single section.
    styles : array_like
        A grouping label defining the plotting style of the markers
        and intervals.
    marker_props : dict
        A dictionary mapping style codes (the values in `styles`) to
        dictionaries defining key/value pairs to be passed as keyword
        arguments to `plot` when plotting markers.  Useful keyword
        arguments are "color", "marker", and "ms" (marker size).
    line_props : dict
        A dictionary mapping style codes (the values in `styles`) to
        dictionaries defining key/value pairs to be passed as keyword
        arguments to `plot` when plotting interval lines.  Useful
        keyword arguments are "color", "linestyle", "solid_capstyle",
        and "linewidth".
    split_names : str
        If not None, this is used to split the values of `lines` into
        substrings that are drawn in the left and right margins,
        respectively.  If None, the values of `lines` are drawn in the
        left margin.
    section_order : array_like
        The section labels in the order in which they appear in the
        dotplot.
    line_order : array_like
        The line labels in the order in which they appear in the
        dotplot.
    stacked : bool
        If True, when multiple points or intervals are drawn on the
        same line, they are offset from each other.
    styles_order : array_like
        If stacked=True, this is the order in which the point styles
        on a given line are drawn from top to bottom (if horizontal
        is True) or from left to right (if horizontal is False).  If
        None (default), the order is lexical.
    striped : bool
        If True, every other line is enclosed in a shaded box.
    horizontal : bool
        If True (default), the lines are drawn horizontally, otherwise
        they are drawn vertically.
    show_names : str
        Determines whether labels (names) are shown in the left and/or
        right margins (top/bottom margins if `horizontal` is True).
        If `both`, labels are drawn in both margins, if 'left', labels
        are drawn in the left or top margin.  If `right`, labels are
        drawn in the right or bottom margin.
    fmt_left_name : callable
        The left/top margin names are passed through this function
        before drawing on the plot.
    fmt_right_name : callable
        The right/bottom marginnames are passed through this function
        before drawing on the plot.
    show_section_titles : bool or None
        If None, section titles are drawn only if there is more than
        one section.  If False/True, section titles are never/always
        drawn, respectively.
    ax : matplotlib.axes
        The axes on which the dotplot is drawn.  If None, a new axes
        is created.

    Returns
    -------
    fig : Figure
        The figure given by `ax.figure` or a new instance.

    Notes
    -----
    `points`, `intervals`, `lines`, `sections`, `styles` must all have
    the same length whenever present.

    References
    ----------
    .. [1] Cleveland, William S. (1993). "Visualizing Data". Hobart Press.
    .. [2] Jacoby, William G. (2006) "The Dot Plot: A Graphical Display
       for Labeled Quantitative Values." The Political Methodologist
       14(1): 6-14.

    Examples
    --------
    This is a simple dotplot with one point per line:

    >>> dot_plot(points=point_values)

    This dotplot has labels on the lines (if elements in
    `label_values` are repeated, the corresponding points appear on
    the same line):

    >>> dot_plot(points=point_values, lines=label_values)
    