from __future__ import annotations
def _mpl_cmdata_to_bokeh_palette(cm_data):
    """Given the data from a Matplotlib colormap as a list of three-item lists in the range 0,1.0,
    convert colors into the range 0,256 and return as a list of tuples"""
    return [(int(r * 256), int(g * 256), int(b * 256)) for r, g, b in cm_data]