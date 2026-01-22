from __future__ import absolute_import
from __future__ import print_function
import webbrowser
from ..palette import Palette
from .colorbrewer_all_schemes import COLOR_MAPS
class BrewerMap(Palette):
    """
    Representation of a colorbrewer2 color map with matplotlib compatible
    views of the map.

    Parameters
    ----------
    name : str
    map_type : str
    colors : list
        Colors as list of 0-255 RGB triplets.

    Attributes
    ----------
    name : str
    type : str
    number : int
        Number of colors in color map.
    colors : list
        Colors as list of 0-255 RGB triplets.
    colorbrewer2_url : str
    hex_colors : list
    mpl_colors : list
    mpl_colormap : matplotlib LinearSegmentedColormap

    """

    @property
    def colorbrewer2_url(self):
        """
        URL that can be used to view the color map at colorbrewer2.org.

        """
        url = 'http://colorbrewer2.org/index.html?type={0}&scheme={1}&n={2}'
        return url.format(self.type.lower(), self.name, self.number)

    def colorbrewer2(self):
        """
        View this color map at colorbrewer2.org. This will open
        colorbrewer2.org in your default web browser.

        """
        webbrowser.open_new_tab(self.colorbrewer2_url)