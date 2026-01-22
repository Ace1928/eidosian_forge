from matplotlib import _api, _docstring
from matplotlib.offsetbox import AnchoredOffsetbox
from matplotlib.patches import Patch, Rectangle
from matplotlib.path import Path
from matplotlib.transforms import Bbox, BboxTransformTo
from matplotlib.transforms import IdentityTransform, TransformedBbox
from . import axes_size as Size
from .parasite_axes import HostAxes
@_api.deprecated('3.8', alternative='Axes.inset_axes')
class InsetPosition:

    @_docstring.dedent_interpd
    def __init__(self, parent, lbwh):
        """
        An object for positioning an inset axes.

        This is created by specifying the normalized coordinates in the axes,
        instead of the figure.

        Parameters
        ----------
        parent : `~matplotlib.axes.Axes`
            Axes to use for normalizing coordinates.

        lbwh : iterable of four floats
            The left edge, bottom edge, width, and height of the inset axes, in
            units of the normalized coordinate of the *parent* axes.

        See Also
        --------
        :meth:`matplotlib.axes.Axes.set_axes_locator`

        Examples
        --------
        The following bounds the inset axes to a box with 20%% of the parent
        axes height and 40%% of the width. The size of the axes specified
        ([0, 0, 1, 1]) ensures that the axes completely fills the bounding box:

        >>> parent_axes = plt.gca()
        >>> ax_ins = plt.axes([0, 0, 1, 1])
        >>> ip = InsetPosition(parent_axes, [0.5, 0.1, 0.4, 0.2])
        >>> ax_ins.set_axes_locator(ip)
        """
        self.parent = parent
        self.lbwh = lbwh

    def __call__(self, ax, renderer):
        bbox_parent = self.parent.get_position(original=False)
        trans = BboxTransformTo(bbox_parent)
        bbox_inset = Bbox.from_bounds(*self.lbwh)
        bb = TransformedBbox(bbox_inset, trans)
        return bb