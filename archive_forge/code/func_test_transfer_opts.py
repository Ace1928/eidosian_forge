import holoviews
import pytest
from holoviews.core import Store
from holoviews.element import Area, Curve
from hvplot.backend_transforms import (
@pytest.mark.parametrize(('element', 'opt', 'val', 'backend', 'opt_kind', 'transf_opt', 'transf_val'), ((Curve([]), 'line_dash', 'dashed', 'matplotlib', 'style', 'linestyle', 'dashed'), (Curve([]), 'line_alpha', 0.123, 'matplotlib', 'style', None, None), (Area([]), 'line_cap', 'square', 'matplotlib', 'style', 'capstyle', 'projecting'), (Curve([]), 'line_dash', 'dashed', 'plotly', 'style', 'dash', 'dash')))
def test_transfer_opts(element, opt, val, backend, opt_kind, transf_opt, transf_val):
    current_backend = Store.current_backend
    if backend not in Store.registry:
        holoviews.extension(backend)
    Store.set_current_backend(backend)
    try:
        element = element.opts(backend='bokeh', **{opt: val})
        new_element = element.apply(_transfer_opts, backend=backend)
        new_opts = new_element.opts.get(opt_kind).kwargs
        if transf_opt is None:
            assert val not in new_opts.values()
        else:
            assert transf_opt in new_opts
            assert new_opts[transf_opt] == transf_val
    finally:
        Store.set_current_backend(current_backend)