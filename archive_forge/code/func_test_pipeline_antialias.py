from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
import datashader as ds
import datashader.transfer_functions as tf
@pytest.mark.parametrize('line_width', [0.0, 0.5, 1.0, 2.0])
def test_pipeline_antialias(line_width):
    glyph = ds.glyphs.LineAxis0('x', 'y')
    glyph.set_line_width(line_width=line_width)
    assert glyph._line_width == line_width
    assert glyph.antialiased == (line_width > 0)
    pipeline = ds.Pipeline(df, glyph)
    img = pipeline(width=cvs10.plot_width, height=cvs10.plot_height, x_range=cvs10.x_range, y_range=cvs10.y_range)
    agg = cvs10.line(df, 'x', 'y', agg=ds.reductions.count(), line_width=line_width)
    assert img.equals(tf.dynspread(tf.shade(agg)))