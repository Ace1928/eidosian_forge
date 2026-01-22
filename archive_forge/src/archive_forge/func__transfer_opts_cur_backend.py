from holoviews.core.overlay import CompositeOverlay
from holoviews.core.options import Store
from holoviews.plotting.util import COLOR_ALIASES
def _transfer_opts_cur_backend(element):
    if Store.current_backend != 'bokeh':
        element = element.apply(_transfer_opts, backend=Store.current_backend)
    return element