from __future__ import absolute_import
from __future__ import print_function
import webbrowser
from ..palette import Palette
class WesAndersonMap(Palette):
    """
    Representation of a color map with matplotlib compatible
    views of the map.

    Parameters
    ----------
    name : str
    map_type : str
    colors : list
        Colors as list of 0-255 RGB triplets.
    url : str
        URL on the web where this color map can be viewed.

    Attributes
    ----------
    name : str
    type : str
    number : int
        Number of colors in color map.
    colors : list
        Colors as list of 0-255 RGB triplets.
    hex_colors : list
    mpl_colors : list
    mpl_colormap : matplotlib LinearSegmentedColormap
    wap_url : str
        URL on the web where this color map can be viewed.

    """

    def __init__(self, name, map_type, colors, url):
        super(WesAndersonMap, self).__init__(name, map_type, colors)
        self.url = url

    def wap(self):
        """
        View this color palette on the web.
        Will open a new tab in your web browser.

        """
        webbrowser.open_new_tab(self.url)