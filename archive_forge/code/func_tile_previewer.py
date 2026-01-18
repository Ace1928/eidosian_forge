from __future__ import annotations
from io import BytesIO
import math
import os
import dask
import dask.bag as db
import numpy as np
from PIL.Image import fromarray
def tile_previewer(full_extent, tileset_url, output_dir=None, filename='index.html', title='Datashader Tileset', min_zoom=0, max_zoom=40, height=None, width=None, **kwargs):
    """Helper function for creating a simple Bokeh figure with
    a WMTS Tile Source.

    Notes
    -----
    - if you don't supply height / width, stretch_both sizing_mode is used.
    - supply an output_dir to write figure to disk.
    """
    try:
        from bokeh.plotting import figure
        from bokeh.models.tiles import WMTSTileSource
        from bokeh.io import output_file, save
        from os import path
    except ImportError:
        raise ImportError('install bokeh to enable creation of simple tile viewer')
    if output_dir:
        output_file(filename=path.join(output_dir, filename), title=title)
    xmin, ymin, xmax, ymax = full_extent
    if height and width:
        p = figure(width=width, height=height, x_range=(xmin, xmax), y_range=(ymin, ymax), tools='pan,wheel_zoom,reset', **kwargs)
    else:
        p = figure(sizing_mode='stretch_both', x_range=(xmin, xmax), y_range=(ymin, ymax), tools='pan,wheel_zoom,reset', **kwargs)
    p.background_fill_color = 'black'
    p.grid.grid_line_alpha = 0
    p.axis.visible = True
    tile_source = WMTSTileSource(url=tileset_url, min_zoom=min_zoom, max_zoom=max_zoom)
    p.add_tile(tile_source, render_parents=False)
    if output_dir:
        save(p)
    return p