from holoviews import Store, extension
from holoviews.core.options import Compositor
from holoviews.operation.element import contours
from ..element import Contours, Polygons
def _load_bokeh():
    from . import bokeh