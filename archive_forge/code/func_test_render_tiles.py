from __future__ import annotations
import datashader as ds
import datashader.transfer_functions as tf
from datashader.colors import viridis
from datashader.tiles import render_tiles
from datashader.tiles import gen_super_tiles
from datashader.tiles import _get_super_tile_min_max
from datashader.tiles import calculate_zoom_level_stats
from datashader.tiles import MercatorTileDefinition
import numpy as np
import pandas as pd
def test_render_tiles():
    full_extent_of_data = (-500000, -500000, 500000, 500000)
    levels = list(range(2))
    output_path = 'test_tiles_output'
    results = render_tiles(full_extent_of_data, levels, load_data_func=mock_load_data_func, rasterize_func=mock_rasterize_func, shader_func=mock_shader_func, post_render_func=mock_post_render_func, output_path=output_path)
    assert results
    assert isinstance(results, dict)
    for level in levels:
        assert level in results
        assert isinstance(results[level], dict)
    assert results[0]['success']
    assert results[0]['stats']
    assert results[0]['supertile_count']