from matplotlib import _api, cbook
import matplotlib.artist as martist
import matplotlib.transforms as mtransforms
from matplotlib.transforms import Bbox
from .mpl_axes import Axes
def _sync_lims(self, parent):
    viewlim = parent.viewLim.frozen()
    mode = self.get_viewlim_mode()
    if mode is None:
        pass
    elif mode == 'equal':
        self.viewLim.set(viewlim)
    elif mode == 'transform':
        self.viewLim.set(viewlim.transformed(self.transAux.inverted()))
    else:
        _api.check_in_list([None, 'equal', 'transform'], mode=mode)