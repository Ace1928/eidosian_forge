from matplotlib import _api, cbook
import matplotlib.artist as martist
import matplotlib.transforms as mtransforms
from matplotlib.transforms import Bbox
from .mpl_axes import Axes
def twin(self, aux_trans=None, axes_class=None):
    """
        Create a twin of Axes with no shared axis.

        While self will have ticks on the left and bottom axis, the returned
        axes will have ticks on the top and right axis.
        """
    if aux_trans is None:
        aux_trans = mtransforms.IdentityTransform()
    ax = self._add_twin_axes(axes_class, aux_transform=aux_trans, viewlim_mode='transform')
    self.axis['top', 'right'].set_visible(False)
    ax.axis['top', 'right'].set_visible(True)
    ax.axis['left', 'bottom'].set_visible(False)
    return ax