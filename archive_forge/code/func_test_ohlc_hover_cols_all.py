import pandas as pd
import hvplot.pandas  # noqa
def test_ohlc_hover_cols_all():
    plot = df.hvplot.ohlc(y=ohlc_cols, hover_cols='all')
    segments = plot.Segments.I
    assert 'Volume' in segments
    tooltips = segments.opts.get('plot').kwargs['tools'][0].tooltips
    assert len(tooltips) == len(df.columns) + 1
    assert tooltips[-1] == ('Volume', '@Volume')