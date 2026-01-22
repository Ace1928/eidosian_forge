from matplotlib import _api, _docstring
from matplotlib.offsetbox import AnchoredOffsetbox
from matplotlib.patches import Patch, Rectangle
from matplotlib.path import Path
from matplotlib.transforms import Bbox, BboxTransformTo
from matplotlib.transforms import IdentityTransform, TransformedBbox
from . import axes_size as Size
from .parasite_axes import HostAxes
class AnchoredLocatorBase(AnchoredOffsetbox):

    def __init__(self, bbox_to_anchor, offsetbox, loc, borderpad=0.5, bbox_transform=None):
        super().__init__(loc, pad=0.0, child=None, borderpad=borderpad, bbox_to_anchor=bbox_to_anchor, bbox_transform=bbox_transform)

    def draw(self, renderer):
        raise RuntimeError('No draw method should be called')

    def __call__(self, ax, renderer):
        if renderer is None:
            renderer = ax.figure._get_renderer()
        self.axes = ax
        bbox = self.get_window_extent(renderer)
        px, py = self.get_offset(bbox.width, bbox.height, 0, 0, renderer)
        bbox_canvas = Bbox.from_bounds(px, py, bbox.width, bbox.height)
        tr = ax.figure.transSubfigure.inverted()
        return TransformedBbox(bbox_canvas, tr)