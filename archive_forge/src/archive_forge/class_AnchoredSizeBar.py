from matplotlib import _api, transforms
from matplotlib.offsetbox import (AnchoredOffsetbox, AuxTransformBox,
from matplotlib.patches import (Rectangle, Ellipse, ArrowStyle,
from matplotlib.text import TextPath
class AnchoredSizeBar(AnchoredOffsetbox):

    def __init__(self, transform, size, label, loc, pad=0.1, borderpad=0.1, sep=2, frameon=True, size_vertical=0, color='black', label_top=False, fontproperties=None, fill_bar=None, **kwargs):
        """
        Draw a horizontal scale bar with a center-aligned label underneath.

        Parameters
        ----------
        transform : `~matplotlib.transforms.Transform`
            The transformation object for the coordinate system in use, i.e.,
            :attr:`matplotlib.axes.Axes.transData`.
        size : float
            Horizontal length of the size bar, given in coordinates of
            *transform*.
        label : str
            Label to display.
        loc : str
            Location of the size bar.  Valid locations are
            'upper left', 'upper center', 'upper right',
            'center left', 'center', 'center right',
            'lower left', 'lower center', 'lower right'.
            For backward compatibility, numeric values are accepted as well.
            See the parameter *loc* of `.Legend` for details.
        pad : float, default: 0.1
            Padding around the label and size bar, in fraction of the font
            size.
        borderpad : float, default: 0.1
            Border padding, in fraction of the font size.
        sep : float, default: 2
            Separation between the label and the size bar, in points.
        frameon : bool, default: True
            If True, draw a box around the horizontal bar and label.
        size_vertical : float, default: 0
            Vertical length of the size bar, given in coordinates of
            *transform*.
        color : str, default: 'black'
            Color for the size bar and label.
        label_top : bool, default: False
            If True, the label will be over the size bar.
        fontproperties : `~matplotlib.font_manager.FontProperties`, optional
            Font properties for the label text.
        fill_bar : bool, optional
            If True and if *size_vertical* is nonzero, the size bar will
            be filled in with the color specified by the size bar.
            Defaults to True if *size_vertical* is greater than
            zero and False otherwise.
        **kwargs
            Keyword arguments forwarded to `.AnchoredOffsetbox`.

        Attributes
        ----------
        size_bar : `~matplotlib.offsetbox.AuxTransformBox`
            Container for the size bar.
        txt_label : `~matplotlib.offsetbox.TextArea`
            Container for the label of the size bar.

        Notes
        -----
        If *prop* is passed as a keyword argument, but *fontproperties* is
        not, then *prop* is assumed to be the intended *fontproperties*.
        Using both *prop* and *fontproperties* is not supported.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from mpl_toolkits.axes_grid1.anchored_artists import (
        ...     AnchoredSizeBar)
        >>> fig, ax = plt.subplots()
        >>> ax.imshow(np.random.random((10, 10)))
        >>> bar = AnchoredSizeBar(ax.transData, 3, '3 data units', 4)
        >>> ax.add_artist(bar)
        >>> fig.show()

        Using all the optional parameters

        >>> import matplotlib.font_manager as fm
        >>> fontprops = fm.FontProperties(size=14, family='monospace')
        >>> bar = AnchoredSizeBar(ax.transData, 3, '3 units', 4, pad=0.5,
        ...                       sep=5, borderpad=0.5, frameon=False,
        ...                       size_vertical=0.5, color='white',
        ...                       fontproperties=fontprops)
        """
        if fill_bar is None:
            fill_bar = size_vertical > 0
        self.size_bar = AuxTransformBox(transform)
        self.size_bar.add_artist(Rectangle((0, 0), size, size_vertical, fill=fill_bar, facecolor=color, edgecolor=color))
        if fontproperties is None and 'prop' in kwargs:
            fontproperties = kwargs.pop('prop')
        if fontproperties is None:
            textprops = {'color': color}
        else:
            textprops = {'color': color, 'fontproperties': fontproperties}
        self.txt_label = TextArea(label, textprops=textprops)
        if label_top:
            _box_children = [self.txt_label, self.size_bar]
        else:
            _box_children = [self.size_bar, self.txt_label]
        self._box = VPacker(children=_box_children, align='center', pad=0, sep=sep)
        super().__init__(loc, pad=pad, borderpad=borderpad, child=self._box, prop=fontproperties, frameon=frameon, **kwargs)