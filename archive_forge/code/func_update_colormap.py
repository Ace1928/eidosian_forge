from ase.gui.i18n import _
import numpy as np
import ase.gui.ui as ui
from ase.gui.utils import get_magmoms
def update_colormap(self, cmap=None, N=26):
    """Called by gui when colormap has changed"""
    if cmap is None:
        cmap = self.cmaps[1].value
    try:
        N = int(self.cmaps[3].value)
    except AttributeError:
        N = 26
    colorscale, mn, mx = self.gui.colormode_data
    if cmap == 'default':
        colorscale = ['#{0:02X}80{0:02X}'.format(int(red)) for red in np.linspace(0, 250, N)]
    elif cmap == 'old':
        colorscale = ['#{0:02X}AA00'.format(int(red)) for red in np.linspace(0, 230, N)]
    else:
        try:
            import pylab as plt
            import matplotlib
            cmap = plt.cm.get_cmap(cmap)
            colorscale = [matplotlib.colors.rgb2hex(c[:3]) for c in cmap(np.linspace(0, 1, N))]
        except (ImportError, ValueError) as e:
            raise RuntimeError('Can not load colormap {0}: {1}'.format(cmap, str(e)))
    self.gui.colormode_data = (colorscale, mn, mx)
    self.gui.draw()