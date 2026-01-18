import colorsys
from itertools import cycle
import numpy as np
import matplotlib as mpl
from .external import husl
from .utils import desaturate, get_color_cycle
from .colors import xkcd_rgb, crayons
from ._compat import get_colormap
def xkcd_palette(colors):
    """Make a palette with color names from the xkcd color survey.

    See xkcd for the full list of colors: https://xkcd.com/color/rgb/

    This is just a simple wrapper around the `seaborn.xkcd_rgb` dictionary.

    Parameters
    ----------
    colors : list of strings
        List of keys in the `seaborn.xkcd_rgb` dictionary.

    Returns
    -------
    palette
        A list of colors as RGB tuples.

    See Also
    --------
    crayon_palette : Make a palette with Crayola crayon colors.

    """
    palette = [xkcd_rgb[name] for name in colors]
    return color_palette(palette, len(palette))