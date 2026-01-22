from holoviews.core.overlay import CompositeOverlay
from holoviews.core.options import Store
from holoviews.plotting.util import COLOR_ALIASES
def _transform_size_to_mpl(width, height, aspect):
    opts = {}
    if width and height:
        opts = {'aspect': width / height, 'fig_size': width / 300.0 * 100}
    elif aspect and width:
        opts = {'aspect': aspect, 'fig_size': width / 300.0 * 100}
    elif aspect and height:
        opts = {'aspect': aspect, 'fig_size': height / 300.0 * 100}
    elif width:
        opts = {'fig_size': width / 300.0 * 100}
    elif height:
        opts = {'fig_size': height / 300.0 * 100}
    return opts