import holoviews
import pytest
from holoviews.core import Store
from holoviews.element import Area, Curve
from hvplot.backend_transforms import (
@pytest.mark.parametrize(('opt', 'val', 'backend', 'opt_kind', 'transf_opt', 'transf_val'), (('line_dash', 'dashed', 'matplotlib', 'style', 'linestyle', 'dashed'), ('line_dash', 'dashed', 'plotly', 'style', 'dash', 'dash')))
def test_transfer_opts_compositeoverlay(opt, val, backend, opt_kind, transf_opt, transf_val):
    current_backend = Store.current_backend
    if backend not in Store.registry:
        holoviews.extension(backend)
    Store.set_current_backend(backend)
    try:
        element = Area([]) * Curve([]).opts(backend='bokeh', **{opt: val})
        new_element = element.apply(_transfer_opts, backend=backend)
        transformed_element = new_element.Curve.I
        new_opts = transformed_element.opts.get(opt_kind).kwargs
        assert transf_opt in new_opts
        assert new_opts[transf_opt] == transf_val
    finally:
        Store.set_current_backend(current_backend)