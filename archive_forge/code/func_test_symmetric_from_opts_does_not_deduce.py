import hvplot
import holoviews as hv
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from holoviews import Store
from holoviews.core.options import Options, OptionTree
def test_symmetric_from_opts_does_not_deduce(self, symmetric_df, backend):
    plot = symmetric_df.hvplot.scatter('x', 'y', c='number', symmetric=False)
    plot_opts = Store.lookup_options(backend, plot, 'plot')
    assert plot_opts.kwargs['symmetric'] is False
    style_opts = Store.lookup_options(backend, plot, 'style')
    assert style_opts.kwargs['cmap'] == 'kbc_r'