import holoviews as hv
import hvplot.pandas  # noqa
import numpy as np
import pandas as pd
import pytest
def test_plot_specific_tiles_doesnt_set_geo(self, simple_df):
    plot = simple_df.hvplot.points('x', 'y', tiles='ESRI')
    assert len(plot) == 2
    assert isinstance(plot.get(0), hv.Tiles)
    assert 'ArcGIS' in plot.get(0).data
    bk_plot = bk_renderer.get_plot(plot)
    assert bk_plot.projection == 'mercator'