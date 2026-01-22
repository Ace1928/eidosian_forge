from matplotlib import _api, transforms
from matplotlib.offsetbox import (AnchoredOffsetbox, AuxTransformBox,
from matplotlib.patches import (Rectangle, Ellipse, ArrowStyle,
from matplotlib.text import TextPath
@_api.deprecated('3.8')
class AnchoredEllipse(AnchoredOffsetbox):

    def __init__(self, transform, width, height, angle, loc, pad=0.1, borderpad=0.1, prop=None, frameon=True, **kwargs):
        """
        Draw an anchored ellipse of a given size.

        Parameters
        ----------
        transform : `~matplotlib.transforms.Transform`
            The transformation object for the coordinate system in use, i.e.,
            :attr:`matplotlib.axes.Axes.transData`.
        width, height : float
            Width and height of the ellipse, given in coordinates of
            *transform*.
        angle : float
            Rotation of the ellipse, in degrees, anti-clockwise.
        loc : str
            Location of the ellipse.  Valid locations are
            'upper left', 'upper center', 'upper right',
            'center left', 'center', 'center right',
            'lower left', 'lower center', 'lower right'.
            For backward compatibility, numeric values are accepted as well.
            See the parameter *loc* of `.Legend` for details.
        pad : float, default: 0.1
            Padding around the ellipse, in fraction of the font size.
        borderpad : float, default: 0.1
            Border padding, in fraction of the font size.
        frameon : bool, default: True
            If True, draw a box around the ellipse.
        prop : `~matplotlib.font_manager.FontProperties`, optional
            Font property used as a reference for paddings.
        **kwargs
            Keyword arguments forwarded to `.AnchoredOffsetbox`.

        Attributes
        ----------
        ellipse : `~matplotlib.patches.Ellipse`
            Ellipse patch drawn.
        """
        self._box = AuxTransformBox(transform)
        self.ellipse = Ellipse((0, 0), width, height, angle=angle)
        self._box.add_artist(self.ellipse)
        super().__init__(loc, pad=pad, borderpad=borderpad, child=self._box, prop=prop, frameon=frameon, **kwargs)