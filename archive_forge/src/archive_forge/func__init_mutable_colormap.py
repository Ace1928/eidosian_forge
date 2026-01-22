import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from .miscplot import palplot
from .palettes import (color_palette, dark_palette, light_palette,
def _init_mutable_colormap():
    """Create a matplotlib colormap that will be updated by the widgets."""
    greys = color_palette('Greys', 256)
    cmap = LinearSegmentedColormap.from_list('interactive', greys)
    cmap._init()
    cmap._set_extremes()
    return cmap