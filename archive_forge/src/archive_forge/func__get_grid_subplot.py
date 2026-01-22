import collections
def _get_grid_subplot(fig, row, col, secondary_y=False):
    try:
        grid_ref = fig._grid_ref
    except AttributeError:
        raise Exception('In order to reference traces by row and column, you must first use plotly.tools.make_subplots to create the figure with a subplot grid.')
    rows = len(grid_ref)
    cols = len(grid_ref[0])
    if not isinstance(row, int) or row < 1 or rows < row:
        raise ValueError('\nThe row argument to get_subplot must be an integer where 1 <= row <= {rows}\n    Received value of type {typ}: {val}'.format(rows=rows, typ=type(row), val=repr(row)))
    if not isinstance(col, int) or col < 1 or cols < col:
        raise ValueError('\nThe col argument to get_subplot must be an integer where 1 <= row <= {cols}\n    Received value of type {typ}: {val}'.format(cols=cols, typ=type(col), val=repr(col)))
    subplot_refs = fig._grid_ref[row - 1][col - 1]
    if not subplot_refs:
        return None
    if secondary_y:
        if len(subplot_refs) > 1:
            layout_keys = subplot_refs[1].layout_keys
        else:
            return None
    else:
        layout_keys = subplot_refs[0].layout_keys
    if len(layout_keys) == 0:
        return SubplotDomain(**subplot_refs[0].trace_kwargs['domain'])
    elif len(layout_keys) == 1:
        return fig.layout[layout_keys[0]]
    elif len(layout_keys) == 2:
        return SubplotXY(xaxis=fig.layout[layout_keys[0]], yaxis=fig.layout[layout_keys[1]])
    else:
        raise ValueError('\nUnexpected subplot type with layout_keys of {}'.format(layout_keys))